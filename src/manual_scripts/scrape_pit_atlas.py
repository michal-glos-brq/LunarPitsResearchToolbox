#!/usr/bin/env python
"""
This script scrapes the Lunar Pits atlas website and directly converts the scraped data
into the final Pydantic-validated format before inserting it into MongoDB.
"""

import os
import sys
import asyncio
import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup
import pandas as pd
import logging
from tqdm import tqdm

from src.db.config import IMG_BASE_FOLDER
from src.db.interface import Sessions
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


async def async_fetch_with_retries(
    session: aiohttp.ClientSession, url: str, REQUEST_MAX_RETRIES: int = 10, pbar: tqdm = None
):
    sleep_time = 10  # Initial delay.
    for _ in range(REQUEST_MAX_RETRIES):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    return response, text
        except Exception as e:
            logging.warning(f"Error fetching {url}: {e}")
        if pbar:
            pbar.set_description(f"Retrying {url} in {sleep_time}s")
        else:
            print(f"Retrying {url} in {sleep_time} seconds.")
        await asyncio.sleep(sleep_time)
        sleep_time = min(sleep_time * 2, 60)
    return None, None


async def async_download_image(session: aiohttp.ClientSession, image_url: str, pbar: tqdm = None):
    sleep_time = 10
    for _ in range(10):  # Fixed retry count.
        try:
            async with session.get(image_url) as response:
                if response.status == 200:
                    content = await response.read()
                    image_name = os.path.basename(image_url)
                    image_path = os.path.join(IMG_BASE_FOLDER, image_name)
                    with open(image_path, "wb") as img_file:
                        img_file.write(content)
                    return image_path
        except Exception as e:
            logging.warning(f"Failed to download image {image_url}: {e}")
        if pbar:
            pbar.set_description(f"Retrying image {image_url} in {sleep_time}s")
        else:
            print(f"Retrying image {image_url} in {sleep_time} seconds.")
        await asyncio.sleep(sleep_time)
        sleep_time = min(sleep_time * 2, 60)
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


def parse_details_and_images(divs, row_name: str):
    """
    Parse the detail page to extract details and image info.
    Note: Image URLs will be downloaded asynchronously later.
    """
    # Parse details table.
    detail_table = divs[0].find("table")
    detail_rows = detail_table.find_all("tr")
    parsed_details = {
        detail.find("th").text.strip().replace(".", "").replace(" ", "_").lower(): detail.find("td").text.strip()
        for detail in detail_rows[1:]
    }
    parsed_details["origin"] = detail_rows[0].find("th").text.strip().split(":")[0].strip()
    parsed_details["name"] = row_name

    # Parse images.
    image_data = []
    images_tables = divs[1].find_all("table")
    for image_table in images_tables:
        image_detail = {"title": image_table.find("th").text.strip(), "object": row_name}
        # Gather image URLs in a list for async download.
        for row in image_table.find_all("tr"):
            if row.find("td"):
                if row.find("th"):
                    key = row.find("th").text.strip().replace(".", "").replace(" ", "_").lower()
                    value = row.find("td").text.strip()
                    image_detail[key] = value
                elif row.find("img"):
                    img = row.find("img")
                    image_url = f"{PIT_ATLAS_BASE_URL}{img['src']}"
                    image_detail.setdefault("image_urls", []).append(image_url)
        image_data.append(image_detail)
    return parsed_details, image_data


def insert_parsed_records(records: List[Dict], collection, model_class):
    """
    Inserts records into the target collection after validating and transforming
    them with the provided Pydantic model. For Pits records, it also adds a 'url'
    field (constructed from PIT_ATLAS_BASE_URL and link_suffix) if available.
    """
    pbar = tqdm(total=len(records), desc=f"Inserting into {collection.name}", dynamic_ncols=True)
    collection.delete_many({})  # Clear existing records.
    parsed_objects = []
    for record in records:
        try:
            # For Pits records, add the full URL if a link_suffix is provided.
            if model_class.__name__ == "PitsMongoObject" and "link_suffix" in record:
                record["url"] = f"{PIT_ATLAS_BASE_URL}{record['link_suffix']}"
            parsed_doc = model_class(**record).dict(by_alias=True)
            # Remove _id so that MongoDB auto-generates it.
            parsed_doc.pop("_id", None)
            parsed_objects.append(parsed_doc)
        except Exception as e:
            print(f"Error processing record {record.get('name', 'unknown')}: {e}")
            continue
        pbar.update(1)
    if parsed_objects:
        collection.insert_many(parsed_objects, ordered=False)
    pbar.close()


async def process_detail_page(row: Dict, session: aiohttp.ClientSession, pbar: tqdm = None):
    detail_url = f"{PIT_ATLAS_BASE_URL}{row['link_suffix']}"
    resp, text = await async_fetch_with_retries(session, detail_url, pbar=pbar)
    if not resp:
        print(f"Failed to fetch details for {row['name']}")
        return None, None
    detail_soup = BeautifulSoup(text, "html.parser")
    divs = detail_soup.find_all("div", {"class": "table-responsive"})
    if len(divs) < 2:
        print(f"Unexpected structure for {row['name']}. Skipping.")
        return None, None
    parsed_details, parsed_images = parse_details_and_images(divs, row["name"])

    # Download images concurrently using the shared session.
    for image_detail in parsed_images:
        if "image_urls" in image_detail:
            tasks = [async_download_image(session, url, pbar=pbar) for url in image_detail["image_urls"]]
            downloaded_paths = await asyncio.gather(*tasks)
            # Use first successfully downloaded image.
            image_detail["image_path"] = next((path for path in downloaded_paths if path), None)
            del image_detail["image_urls"]
    return parsed_details, parsed_images


async def scrape_and_parse_lunar_pits():
    os.makedirs(IMG_BASE_FOLDER, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        # Fetch list page.
        resp, text = await async_fetch_with_retries(session, PIT_ATLAS_LIST_URL)
        if not resp:
            print("Failed to fetch the list page.")
            sys.exit(1)
        soup = BeautifulSoup(text, "html.parser")
        table = soup.find("table", {"id": "pitsTable"})
        headers = parse_table_headers(table)
        if headers != EXPECTED_PIT_TABLE_COLUMNS:
            print(f"Unexpected table headers: {headers}")
            raise ValueError("Table headers have changed. Please update the model.")
        rows_data = parse_table_rows(table)
        general_df = pd.DataFrame(rows_data, columns=PIT_TABLE_COLUMNS)

        detail_data = []
        image_data = []
        tasks = []
        pbar_details = tqdm(
            total=general_df.shape[0], desc="Fetching Details", dynamic_ncols=True, leave=True, file=sys.stderr
        )
        # Create async tasks for each detail page, sharing the session.
        for _, row in general_df.iterrows():
            tasks.append(process_detail_page(row, session, pbar=pbar_details))
        results = await asyncio.gather(*tasks)
        for res in results:
            if res is None:
                continue
            parsed_details, parsed_images = res
            if parsed_details:
                detail_data.append(parsed_details)
            if parsed_images:
                image_data.extend(parsed_images)
            pbar_details.update(1)
        pbar_details.close()

        # Convert DataFrames.
        general_records = general_df.to_dict(orient="records")
        detail_records = pd.DataFrame(detail_data).to_dict(orient="records")
        image_records = pd.DataFrame(image_data).to_dict(orient="records")

        # Insert records into MongoDB.
        PitsCollectionOut, PitDetailsCollectionOut, ImageCollectionOut = Sessions.get_lunar_pit_collections()
        print("Inserting PITS ...")
        insert_parsed_records(general_records, PitsCollectionOut, PitsMongoObject)
        print("Inserting PIT DETAILS ...")
        insert_parsed_records(detail_records, PitDetailsCollectionOut, PitDetailsMongoObject)
        print("Inserting IMAGES ...")
        insert_parsed_records(image_records, ImageCollectionOut, ImageMongoObject)
        print("Done.")


if __name__ == "__main__":
    asyncio.run(scrape_and_parse_lunar_pits())
