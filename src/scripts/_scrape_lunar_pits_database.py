"""
This code scrapes the Lunar Pits database and saves it into a MongoDB database.
"""

import os
import sys
import requests
from time import sleep

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from src.db.config import IMG_BASE_FOLDER
from src.db.interface import Sessions
from src.SPICE.config import HEADERS as BROWSER_HEADERS

# It's single use constants, no need to put them out in config
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
    """Fetch a URL with retry logic."""
    sleep_time = 10 # arbitrary
    for _ in range(REQUEST_MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response
        except:
            if pbar:
                pbar.set_description(f"Retrying - {sleep_time} s")
            else:
                print(f"Retrying in {sleep_time} seconds.")
            sleep(sleep_time)
            sleep_time *= 2
            sleep_time = min(sleep_time, 60)  # Cap sleep time to 60 seconds
    return None


def parse_table_headers(table):
    """
    Extract headers from a table.
    """
    return [header.text.strip() for header in table.find("thead").find_all("th")]


def parse_table_rows(table):
    """Extract rows of data from a table."""
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
    """Parse details and images from the detail page."""
    # Parse details table
    detail_table = divs[0].find("table")
    detail_rows = detail_table.find_all("tr")
    parsed_details = {
        detail.find("th").text.strip().replace(".", "").replace(" ", "_").lower(): detail.find("td").text.strip()
        for detail in detail_rows[1:]
    }
    parsed_details["origin"] = detail_rows[0].find("th").text.strip().split(":")[0].strip()
    parsed_details["name"] = row_name

    # Parse images and related metadata
    image_data = []
    images_tables = divs[1].find_all("table")
    for image_table in images_tables:
        image_detail = {"title": image_table.find("th").text.strip(), "object": row_name}
        for dato in image_table.find_all("tr"):
            if dato.find("td"):
                if dato.find("th"):
                    key = dato.find("th").text.strip().replace(".", "").replace(" ", "_").lower()
                    value = dato.find("td").text.strip()
                    image_detail[key] = value
                elif img := dato.find("img"):
                    image_url = f"{PIT_ATLAS_BASE_URL}{img['src']}"
                    image_path = download_image(image_url, pbar=pbar)
                    if image_path:
                        image_detail["image_path"] = image_path
        image_data.append(image_detail)
    return parsed_details, image_data


# Main Script
def scrape_lunar_pit_atlas():

    def replace_collection(collection, df):
        """Rewrites a collection by dataframe data"""
        # Drop the existing collection
        collection.drop()
        # Insert the new DataFrame
        records = df.to_dict(orient="records")  # Convert DataFrame to list of dicts
        collection.insert_many(records)
        print(f"Replaced collection '{collection.full_name}' with {len(records)} records.")

    # Ensure image folder exists
    os.makedirs(IMG_BASE_FOLDER, exist_ok=True)

    # Fetch main list page
    response = requests.get(PIT_ATLAS_LIST_URL, headers=BROWSER_HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch the list page. Status code: {response.status_code}")
        exit()

    # Parse main page for pit objects
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"id": "pitsTable"})
    headers = parse_table_headers(table)
    if headers != EXPECTED_PIT_TABLE_COLUMNS:
        raise ValueError("Table headers have changed. Please update the script.")

    rows_data = parse_table_rows(table)
    general_df = pd.DataFrame(rows_data, columns=PIT_TABLE_COLUMNS)

    detail_data = []
    image_data = []

    # Fetch detailed pages and parse
    pbar = tqdm(total=general_df.shape[0], desc="Fetching Details", dynamic_ncols=True, leave=True, file=sys.stderr)
    for _, row in general_df.iterrows():
        pbar.set_description("Fetching Details ...")
        detail_url = f'{PIT_ATLAS_BASE_URL}{row["link_suffix"]}'
        detail_response = fetch_with_retries(detail_url, BROWSER_HEADERS, pbar=pbar)
        if not detail_response:
            print(f"Failed to fetch details for {row['name']}")
            continue

        detail_soup = BeautifulSoup(detail_response.content, "html.parser")
        divs = detail_soup.find_all("div", {"class": "table-responsive"})
        if len(divs) < 2:
            print(f"Unexpected structure for {row['name']}. Skipping.")
            continue

        parsed_details, parsed_images = parse_details_and_images(divs, row["name"], pbar=pbar)
        detail_data.append(parsed_details)
        image_data.extend(parsed_images)
        pbar.update(1)

    pbar.close()

    # Convert to DataFrames
    detail_df = pd.DataFrame(detail_data)
    image_df = pd.DataFrame(image_data)

    # Raw data
    pit_collection, pit_detail_collection, image_collection = Sessions.get_lunar_pit_collections(parsed=False)

    # Replace collections in MongoDB
    print("Replacing PITS collection in MongoDB ...")
    replace_collection(pit_collection, general_df)
    print("Replacing PIT_DETAILS and IMAGES collections in MongoDB ...")
    replace_collection(pit_detail_collection, detail_df)
    print("Replacing IMAGES collection in MongoDB ...")
    replace_collection(image_collection, image_df)
    print("Done.")


if __name__ == "__main__":
    scrape_lunar_pit_atlas()
