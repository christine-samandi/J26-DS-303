
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient


# Locate the local .env file.
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

mongodb_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("MONGODB_DATABASE", "c4_assurance")

if not mongodb_uri:
    raise RuntimeError("MONGODB_URI is missing from .env")

client = MongoClient(
    mongodb_uri,
    serverSelectionTimeoutMS=10000,
)

try:
    # Check whether MongoDB Atlas is reachable.
    client.admin.command("ping")

    database = client[database_name]

    print("MongoDB Atlas connection successful!")
    print(f"Database selected: {database.name}")

except Exception as error:
    print("MongoDB connection failed.")
    print(f"Error type: {type(error).__name__}")
    print("Check credentials, network access, and connection settings.")

finally:
    client.close()
