"""
This script will parse the existing Lunar Pit MongoDB database and create a new parsed MongoDB database

Requires to run mongo.scrape-lunar-pits-database.py to scrape the dataset first
"""

import sys
from tqdm import tqdm

from src.db.models.lunar_pit_atlas import PitDetailsMongoObject, PitsMongoObject, ImageMongoObject
from src.db.interface import Sessions


def perform_largescale_conversion_with_pydantic(collection_in, collection_out, model_class):
    """
    Perform large-scale transformation using Pydantic models for validation and conversion.
    """
    pbar = tqdm(
        total=collection_in.count_documents({}),
        desc=f"Processing {collection_in.name}",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )
    for doc in collection_in.find({}):
        try:
            # Validate and transform using Pydantic
            transformed_doc = model_class(**doc).dict(by_alias=True)
            # Exclude `_id` from the update
            update_fields = {k: v for k, v in transformed_doc.items() if k != "_id"}
        except Exception as e:
            print(f"Error processing document {doc['_id']}: {e}")
            continue  # Skip to the next document on error

        # Update the document in the target collection
        collection_out.update_one(
            {"_id": doc["_id"]},  # Match by unique ID
            {"$set": update_fields},
            upsert=True,  # Ensures the document is inserted if not already present
        )
        pbar.update(1)
    pbar.close()


def parse_lunar_pits_db():
    PitsCollectionOut, PitDetailsCollectionOut, ImageCollectionOut = Sessions.get_lunar_pit_collections()
    PitsCollectionIn, PitDetailsCollectionIn, ImageCollectionIn = Sessions.get_lunar_pit_collections(parsed=True)

    # Make sure to start with clean colletions. The script can upsert, but fresh start is better, considering the size of the database
    PitsCollectionIn.delete_many({})
    PitDetailsCollectionIn.delete_many({})
    ImageCollectionIn.delete_many({})

    perform_largescale_conversion_with_pydantic(ImageCollectionIn, ImageCollectionOut, ImageMongoObject)

    perform_largescale_conversion_with_pydantic(PitDetailsCollectionIn, PitDetailsCollectionOut, PitDetailsMongoObject)

    perform_largescale_conversion_with_pydantic(PitsCollectionIn, PitsCollectionOut, PitsMongoObject)

if __name__ == "__main__":
    parse_lunar_pits_db()
