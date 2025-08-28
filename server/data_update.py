import json
from pymongo import MongoClient

# === CONFIGURATION ===
MONGO_URI = "mongodb+srv://user3:12345@projectalexdb.noggc5o.mongodb.net/"  # Change if using Atlas
DATABASE_NAME = "ProjectAlex"
COLLECTION_NAME = "user3"
JSON_FILE = "test.json"

# === CONNECT TO MONGO ===
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# === LOAD JSON DATA ===
with open(JSON_FILE, 'r') as file:
    data = json.load(file)
print(data)
# === INSERT DATA ===
if isinstance(data, list):
    result = collection.insert_many(data)
    print(f"{len(result.inserted_ids)} documents inserted.")
elif isinstance(data, dict):
    result = collection.insert_one(data)
    print("1 document inserted.")
else:
    print("Invalid JSON format.")

# === CLOSE CONNECTION ===
client.close()
