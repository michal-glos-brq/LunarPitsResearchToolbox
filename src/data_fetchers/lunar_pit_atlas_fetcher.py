"""
This script scrapes the Lunar Pit Atlas data, processes it using Pydantic models,
and saves only the parsed records into the parsed MongoDB database.
"""

import os
import sys
import requests
from time import sleep

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient, GEOSPHERE

from src.db.models.lunar_pit_atlas import ImageCollection, PitDetailsCollection, PitsCollection
from src.db.config import (
    IMG_BASE_FOLDER,
    MONGO_URI,
    PIT_ATLAS_PARSED_DB_NAME,
    PIT_COLLECTION_NAME,
    PIT_DETAIL_COLLECTION_NAME,
    PIT_ATLAS_IMAGE_COLLECTION_NAME,
)


PIT_ATLAS_LIST_URL = "https://www.lroc.asu.edu/atlases/pits/list"
PIT_ATLAS_BASE_URL = "https://www.lroc.asu.edu/"

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.170 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

REQUEST_MAX_RETRIES = 5
INITIAL_DOWNLOAD_RESET_TIME_SECONDS = 15


EXPECTED_PIT_TABLE_COLUMNS = [
    "Host Feat.",
    "Name",
    "Lat.",
    "Long.",
    "Funnel Max Diam. (m)",
    "Funnel Min Diam. (m)",
    "Inner Max Diam. (m)",
    "Inner Max Diam. Sorting",
    "Inner Min Diam. (m)",
    "Inner Min Diam. Sorting",
    "Azimuth",
    "Depth (m)",
    "Depth Sorting",
]

PIT_TABLE_COLUMNS = [
    "hosting_feature",
    "name",
    "latitude",
    "longitude",
    "funnel_max_diameter",
    "funnel_min_diameter",
    "inner_max_diameter",
    "inner_max_diameter_sorting",
    "inner_min_diameter",
    "inner_min_diameter_sorting",
    "azimuth",
    "depth",
    "depth_sorting",
    "link_suffix",
]


def fetch_with_retries(url, headers, max_retries=REQUEST_MAX_RETRIES, pbar=None):
    sleep_time = INITIAL_DOWNLOAD_RESET_TIME_SECONDS
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response
        except Exception as e:
            if pbar:
                pbar.set_description(f"Retry {attempt+1}/{max_retries} in {sleep_time} s")
            else:
                print(f"Error fetching {url}: {e}. Retrying in {sleep_time} seconds.")
        sleep(sleep_time)
        sleep_time *= 2
    print(f"Failed to fetch {url} after {max_retries} attempts.")
    return None


def parse_table_headers(table):
    thead = table.find("thead")
    if not thead:
        raise ValueError("No <thead> found in table")
    return [th.text.strip() for th in thead.find_all("th")]


def parse_table_rows(table):
    tbody = table.find("tbody")
    if not tbody:
        raise ValueError("No <tbody> found in table")
    rows = tbody.find_all("tr")
    data = []
    for row in rows:
        cells = row.find_all("td")
        cell_data = []
        object_link = None
        for cell in cells:
            link = cell.find("a")
            if link:
                cell_data.append(cell.text.strip())
                object_link = link.get("href")
            else:
                cell_data.append(cell.text.strip())
        cell_data.append(object_link)
        data.append(cell_data)
    return data


def download_image(image_url, pbar=None):
    img_response = fetch_with_retries(image_url, BROWSER_HEADERS, pbar=pbar)
    if img_response:
        image_name = os.path.basename(image_url)
        image_path = os.path.join(IMG_BASE_FOLDER, image_name)
        try:
            with open(image_path, "wb") as img_file:
                img_file.write(img_response.content)
            return image_path
        except Exception as e:
            print(f"Error saving image {image_url}: {e}")
            return None
    return None


def parse_details_and_images(divs, row_name, pbar=None):
    # Parse details table
    detail_table = divs[0].find("table")
    if not detail_table:
        print(f"Warning: No details table found for {row_name}")
        return {}, []
    detail_rows = detail_table.find_all("tr")
    if not detail_rows:
        print(f"Warning: No rows in details table for {row_name}")
        return {}, []
    # Assume the first row contains the origin info
    parsed_details = {
        detail.find("th").text.strip().replace(".", "").replace(" ", "_").lower(): detail.find("td").text.strip()
        for detail in detail_rows[1:] if detail.find("th") and detail.find("td")
    }
    origin_text = detail_rows[0].find("th").text.strip() if detail_rows[0].find("th") else ""
    parsed_details["origin"] = origin_text.split(":")[0].strip() if ":" in origin_text else origin_text
    parsed_details["name"] = row_name

    # Parse images tables
    image_data = []
    images_tables = divs[1].find_all("table")
    for image_table in images_tables:
        title_el = image_table.find("th")
        if not title_el:
            continue
        image_detail = {"title": title_el.text.strip(), "object": row_name}
        for row in image_table.find_all("tr"):
            if row.find("td"):
                if row.find("th"):
                    key = row.find("th").text.strip().replace(".", "").replace(" ", "_").lower()
                    value = row.find("td").text.strip()
                    image_detail[key] = value
                elif (img := row.find("img")) is not None:
                    image_url = f"{PIT_ATLAS_BASE_URL}{img.get('src')}"
                    image_path = download_image(image_url, pbar=pbar)
                    if image_path:
                        image_detail["image_path"] = image_path
        # Only append image_detail if image_path was successfully downloaded.
        if "image_path" in image_detail:
            image_data.append(image_detail)
        else:
            print(f"Warning: No image downloaded for {row_name} in table titled '{image_detail.get('title')}'")
    return parsed_details, image_data


def scrape_and_process():
    os.makedirs(IMG_BASE_FOLDER, exist_ok=True)
    response = requests.get(PIT_ATLAS_LIST_URL, headers=BROWSER_HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch list page. Status code: {response.status_code}")
        sys.exit(1)

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "pitsTable"})
    if table is None:
        print("Error: Could not find pitsTable in the page.")
        sys.exit(1)

    headers = parse_table_headers(table)
    if headers != EXPECTED_PIT_TABLE_COLUMNS:
        raise ValueError("Table headers have changed. Please update the script.")

    rows_data = parse_table_rows(table)
    general_df = pd.DataFrame(rows_data, columns=PIT_TABLE_COLUMNS)

    detail_list = []
    image_list = []

    pbar = tqdm(total=general_df.shape[0], desc="Fetching Details", dynamic_ncols=True, leave=True, file=sys.stderr)
    for _, row in general_df.iterrows():
        detail_url = f"{PIT_ATLAS_BASE_URL}{row['link_suffix']}"
        detail_response = fetch_with_retries(detail_url, BROWSER_HEADERS, pbar=pbar)
        if not detail_response:
            print(f"Failed to fetch details for {row['name']}")
            pbar.update(1)
            continue

        detail_soup = BeautifulSoup(detail_response.content, "html.parser")
        divs = detail_soup.find_all("div", {"class": "table-responsive"})
        if len(divs) < 2:
            print(f"Unexpected structure for {row['name']}. Skipping details.")
            pbar.update(1)
            continue

        parsed_details, parsed_images = parse_details_and_images(divs, row["name"], pbar=pbar)
        if parsed_details:
            detail_list.append(parsed_details)
        if parsed_images:
            image_list.extend(parsed_images)
        pbar.update(1)
    pbar.close()

    return general_df.to_dict(orient="records"), detail_list, image_list


def process_and_insert(collection_data, model_class, collection_out):
    processed = []
    for doc in tqdm(collection_data, desc=f"Processing {collection_out.name}", dynamic_ncols=True):
        try:
            processed_doc = model_class(**doc).dict(by_alias=True)
            processed.append(processed_doc)
        except Exception as e:
            print(f"Error processing document {doc.get('name', doc.get('_id', ''))}: {e}")
    if processed:
        collection_out.insert_many(processed)
        print(f"Inserted {len(processed)} records into {collection_out.name}.")
    else:
        print(f"No valid records to insert into {collection_out.name}.")


def main():
    # Scrape data and process details/images
    general_data, detail_data, image_data = scrape_and_process()

    # Connect to MongoDB and target parsed DB
    client = MongoClient(MONGO_URI)
    db_out = client[PIT_ATLAS_PARSED_DB_NAME]

    # Process and insert Pits collection
    if PIT_COLLECTION_NAME in db_out.list_collection_names():
        db_out.drop_collection(PIT_COLLECTION_NAME)
    pits_collection = db_out.create_collection(PIT_COLLECTION_NAME)
    pits_collection.create_index([("location", GEOSPHERE)])
    pits_collection.create_index([("name", 1)])
    process_and_insert(general_data, PitsCollection, pits_collection)

    # Process and insert Pit Details collection
    if PIT_DETAIL_COLLECTION_NAME in db_out.list_collection_names():
        db_out.drop_collection(PIT_DETAIL_COLLECTION_NAME)
    pit_details_collection = db_out.create_collection(PIT_DETAIL_COLLECTION_NAME)
    pit_details_collection.create_index([("location", GEOSPHERE)])
    pit_details_collection.create_index([("name", 1)])
    process_and_insert(detail_data, PitDetailsCollection, pit_details_collection)

    # Process and insert Images collection
    if PIT_ATLAS_IMAGE_COLLECTION_NAME in db_out.list_collection_names():
        db_out.drop_collection(PIT_ATLAS_IMAGE_COLLECTION_NAME)
    images_collection = db_out.create_collection(PIT_ATLAS_IMAGE_COLLECTION_NAME)
    process_and_insert(image_data, ImageCollection, images_collection)

    print("Done.")


if __name__ == "__main__":
    main()
