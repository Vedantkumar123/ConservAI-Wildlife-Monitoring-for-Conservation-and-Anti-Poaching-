from pymongo import MongoClient
import pandas as pd

# === CONFIGURATION ===
MONGO_URI = "mongodb+srv://user3:12345@projectalexdb.noggc5o.mongodb.net/ProjectAlex?retryWrites=true&w=majority"
DATABASE_NAME = "ProjectAlex"
COLLECTION_NAME = "user3"
EXCEL_FILE = "detections_export.xlsx"

# === CONNECT TO MONGO ===
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# === FETCH DOCUMENTS ===
cursor = collection.find()
data = list(cursor)
# === REMOVE '_id' FIELD (optional) ===
for doc in data:
    doc.pop("_id", None)  # remove ObjectId if not needed in Excel

# === CONVERT TO EXCEL ===
df = pd.DataFrame(data)
df.to_excel(EXCEL_FILE, index=False)

print(f"Exported {len(data)} records to '{EXCEL_FILE}'")

# === CLOSE CONNECTION ===
client.close()
