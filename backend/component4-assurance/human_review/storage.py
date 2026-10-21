
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

from human_review.mongo_repository import MongoReviewCaseRepository
from human_review.repository import ReviewCaseRepository


def build_review_repository():
    """
    Select the repository backend.

    memory  = local in-memory repository
    mongodb = persistent MongoDB Atlas repository
    """

    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    storage = os.getenv(
        "C4_REVIEW_STORAGE", "memory"
    ).strip().lower()

    if storage == "memory":
        return ReviewCaseRepository()

    if storage != "mongodb":
        raise ValueError(
            f"Unsupported review storage backend: {storage}"
        )

    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv(
        "MONGODB_DATABASE", "c4_assurance"
    )

    if not mongodb_uri:
        raise RuntimeError(
            "MONGODB_URI is required for MongoDB storage."
        )

    client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=10000,
    )

    # Verify connection before using the database.
    client.admin.command("ping")

    collection = client[database_name]["review_cases"]

    return MongoReviewCaseRepository(collection)
