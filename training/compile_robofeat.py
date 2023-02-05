import pickle
import pymongo
import json
from tqdm import tqdm
import pandas as pd

collection_name = "matbench-datasets"
database_name = "robocrystallographer"
dataSource = "Cluster0"
app_name = "data-cqjnk"
cluster_uri = "cluster0.fyeompa"

with open("training/secrets.json", "r") as f:
    secrets = json.load(f)

username = secrets["PYMONGO_USERNAME"]
password = secrets["PYMONGO_PASSWORD"]

client = pymongo.MongoClient(
    f"mongodb+srv://{username}:{password}@{cluster_uri}.mongodb.net/?retryWrites=true&w=majority"  # noqa: E501
)
db = client[database_name]
collection = db[collection_name]

posts = collection.find({"cif": {"$exists": True}})
results = [post for post in tqdm(posts)]

df = pd.DataFrame(results)

with open("matbench_datasets/robofeat.pkl", "wb") as f:
    pickle.dump(df, f)

df.to_csv("matbench_datasets/robofeat.csv")

1 + 1
