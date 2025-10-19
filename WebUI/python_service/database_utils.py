# database_utils.py

import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import json

# --- Constants ---
# It's good practice to use environment variables for sensitive data like the MONGO_URI
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://saptarshiacharya33_db_user:pzxGiZWUB0Jkl4Fh@cluster0.ituepua.mongodb.net/")
DATABASE_NAME = "ProjectALex"
COLLECTION_NAME = "ALEX"
# Cooldown period in seconds to prevent duplicate entries for the same detection
DUPLICATE_THRESHOLD_SECONDS = 300  # 5 minutes

# --- MongoDB Client Initialization ---
# Initialize the client once and reuse the connection
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    # The ismaster command is cheap and does not require auth, used to verify connection.
    client.admin.command('ismaster')
    print("✅ Successfully connected to MongoDB.")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    client = None
    db = None
    collection = None

def check_for_recent_duplicate(detection_data: dict) -> bool:
    """
    Checks if a similar detection has been logged recently.

    A detection is a duplicate if the same Animal, Cam_id, and Location
    are found within the DUPLICATE_THRESHOLD_SECONDS.
    """
    # CORRECTED CHECK: Use 'is None' instead of 'not collection'
    if collection is None:
        print("⚠️ MongoDB collection not available. Skipping duplicate check.")
        return False

    # Calculate the time threshold for checking duplicates
    threshold_time = datetime.utcnow() - timedelta(seconds=DUPLICATE_THRESHOLD_SECONDS)

    # Query for a document that matches the key criteria and is recent
    query = {
        "Animal": detection_data.get("Animal"),
        "Cam_id": detection_data.get("Cam_id"),
        "Location": detection_data.get("Location"),
        "Time": {"$gte": threshold_time.isoformat()}
    }

    duplicate = collection.find_one(query)
    
    # If a document is found, it's a duplicate
    return duplicate is not None

def save_detection(detection_data: dict):
    """
    Saves a detection to the database after checking for duplicates.
    """
    # CORRECTED CHECK: Use 'is None' instead of 'not collection'
    if collection is None:
        print("⚠️ MongoDB collection not available. Cannot save detection.")
        return

    # 1. Check for duplicates before inserting
    if check_for_recent_duplicate(detection_data):
        print(f"🔄 Duplicate detection skipped for '{detection_data.get('Animal')}' from Cam '{detection_data.get('Cam_id')}'.")
        return

    try:
        # 2. Add a new ObjectId and ensure time is in ISO format string
        detection_data_to_insert = detection_data.copy()
        detection_data_to_insert["_id"] = ObjectId()
        detection_data_to_insert["Time"] = datetime.utcnow().isoformat()
        
        # 3. Insert the new record into MongoDB
        result = collection.insert_one(detection_data_to_insert)

        # 4. Log the successful insertion
        print("\n📌 New Detection Inserted into MongoDB:")
        print(f"   Document ID: {result.inserted_id}")
        print(json.dumps(detection_data_to_insert, indent=4, default=str))

    except Exception as e:
        print(f"❌ Error saving detection to MongoDB: {e}")