#!/usr/bin/env python
"""
This script scrapes the Lunar Pits atlas website and directly converts the scraped data
into the final Pydantic-validated format before inserting it into MongoDB.
"""

import os
import sys
from time import sleep
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from src.db.config import IMG_BASE_FOLDER
from src.db.interface import Sessions
from src.SPICE.config import HEADERS as BROWSER_HEADERS
from src.db.models.lunar_pit_atlas import PitDetailsMongoObject, PitsMongoObject, ImageMongoObject

# Constants and configuration.
PIT_ATLAS_LIST_URL = "https://www.lroc.asu.edu/atlases/pits/list"
PIT_ATLAS_BASE_URL = "https://www.lroc.asu.edu/"

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


def fetch_with_retries(url, headers, REQUEST_MAX_RETRIES=10, pbar=None):
    sleep_time = 10  # Initial delay.
    for _ in range(REQUEST_MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response
        except Exception:
            pass
        if pbar:
            pbar.set_description(f"Retrying in {sleep_time}s")
        else:
            print(f"Retrying in {sleep_time} seconds.")
        sleep(sleep_time)
        sleep_time = min(sleep_time * 2, 60)  # Cap the sleep time to 60 seconds.
    return None


def parse_table_headers(table):
    """Extract table headers."""
    return [header.text.strip() for header in table.find("thead").find_all("th")]


def parse_table_rows(table):
    """Extract rows of data from the table."""
    rows = table.find("tbody").find_all("tr")
    data = []
    for row in rows:
        cells = row.find_all("td")
        cell_data = []
        object_link = None
        for cell in cells:
            link = cell.find("a")
            if link:
                cell_data.append(cell.text.strip())
                object_link = link["href"]
            else:
                cell_data.append(cell.text.strip())
        cell_data.append(object_link)
        data.append(cell_data)
    return data


def download_image(image_url, pbar=None):
    """Download an image and save it locally."""
    img_response = fetch_with_retries(image_url, BROWSER_HEADERS, pbar=pbar)
    if img_response:
        image_name = os.path.basename(image_url)
        image_path = os.path.join(IMG_BASE_FOLDER, image_name)
        with open(image_path, "wb") as img_file:
            img_file.write(img_response.content)
        return image_path
    return None


def parse_details_and_images(divs, row_name, pbar=None):
    """
    Parse the detail page:
      - Extract detail information from the first table.
      - Extract image details from the second table.
    """
    # Parse details table.
    detail_table = divs[0].find("table")
    detail_rows = detail_table.find_all("tr")
    parsed_details = {
        detail.find("th").text.strip().replace(".", "").replace(" ", "_").lower():
            detail.find("td").text.strip()
        for detail in detail_rows[1:]
    }
    parsed_details["origin"] = detail_rows[0].find("th").text.strip().split(":")[0].strip()
    parsed_details["name"] = row_name

    # Parse images.
    image_data = []
    images_tables = divs[1].find_all("table")
    for image_table in images_tables:
        image_detail = {"title": image_table.find("th").text.strip(), "object": row_name}
        for row in image_table.find_all("tr"):
            if row.find("td"):
                if row.find("th"):
                    key = row.find("th").text.strip().replace(".", "").replace(" ", "_").lower()
                    value = row.find("td").text.strip()
                    image_detail[key] = value
                elif img := row.find("img"):
                    image_url = f"{PIT_ATLAS_BASE_URL}{img['src']}"
                    image_path = download_image(image_url, pbar=pbar)
                    if image_path:
                        image_detail["image_path"] = image_path
        image_data.append(image_detail)
    return parsed_details, image_data


def insert_parsed_records(records: List[Dict], collection, model_class):
    """
    Insert records into the target collection after validating and transforming
    them with the provided Pydantic model.
    """
    pbar = tqdm(total=len(records), desc=f"Inserting into {collection.name}", dynamic_ncols=True)
    for record in records:
        try:
            parsed_doc = model_class(**record).dict(by_alias=True)
            # Exclude _id if present.
            if "_id" in parsed_doc:
                del parsed_doc["_id"]
        except Exception as e:
            print(f"Error processing record {record.get('_id', 'unknown')}: {e}")
            continue
        collection.update_one({"_id": record.get("_id")}, {"$set": parsed_doc}, upsert=True)
        pbar.update(1)
    pbar.close()


def scrape_and_parse_lunar_pits():
    # Ensure image folder exists.
    os.makedirs(IMG_BASE_FOLDER, exist_ok=True)
    response = requests.get(PIT_ATLAS_LIST_URL, headers=BROWSER_HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch the list page. Status code: {response.status_code}")
        sys.exit(1)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "pitsTable"})
    headers = parse_table_headers(table)
    if headers != EXPECTED_PIT_TABLE_COLUMNS:
        print(f"Unexpected table headers: {headers}")
        raise ValueError("Table headers have changed. Please update the model.")
    rows_data = parse_table_rows(table)
    general_df = pd.DataFrame(rows_data, columns=PIT_TABLE_COLUMNS)

    detail_data = []
    image_data = []
    pbar_details = tqdm(total=general_df.shape[0], desc="Fetching Details", dynamic_ncols=True, leave=True, file=sys.stderr)
    for _, row in general_df.iterrows():
        pbar_details.set_description("Fetching Details ...")
        detail_url = f"{PIT_ATLAS_BASE_URL}{row['link_suffix']}"
        detail_response = fetch_with_retries(detail_url, BROWSER_HEADERS, pbar=pbar_details)
        if not detail_response:
            print(f"Failed to fetch details for {row['name']}")
            continue
        detail_soup = BeautifulSoup(detail_response.content, "html.parser")
        divs = detail_soup.find_all("div", {"class": "table-responsive"})
        if len(divs) < 2:
            print(f"Unexpected structure for {row['name']}. Skipping.")
            continue
        parsed_details, parsed_images = parse_details_and_images(divs, row["name"], pbar=pbar_details)
        detail_data.append(parsed_details)
        image_data.extend(parsed_images)
        pbar_details.update(1)
    pbar_details.close()

    # Convert DataFrames.
    general_records = general_df.to_dict(orient="records")
    detail_records = pd.DataFrame(detail_data).to_dict(orient="records")
    image_records = pd.DataFrame(image_data).to_dict(orient="records")

    # Get final MongoDB collections (final parsed collections).
    PitsCollectionOut, PitDetailsCollectionOut, ImageCollectionOut = Sessions.get_lunar_pit_collections()

    print("Inserting PITS ...")
    insert_parsed_records(general_records, PitsCollectionOut, PitsMongoObject)
    print("Inserting PIT DETAILS ...")
    insert_parsed_records(detail_records, PitDetailsCollectionOut, PitDetailsMongoObject)
    print("Inserting IMAGES ...")
    insert_parsed_records(image_records, ImageCollectionOut, ImageMongoObject)
    print("Done.")


if __name__ == "__main__":
    # Import BeautifulSoup here to avoid circular imports if needed.
    from bs4 import BeautifulSoup
    scrape_and_parse_lunar_pits()
