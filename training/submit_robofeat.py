import os
import random
from time import time
from uuid import uuid4
import pandas as pd
import submitit
import dill as pickle
from robocrys.featurize.featurizer import RobocrysFeaturizer
from tqdm import tqdm
from pymatgen.core import Structure
from datetime import datetime
import json
import requests
import ray
import pymongo

import numpy as np

#########################
# User-defined variables
#########################
dummy = False
collection_name = "matbench-datasets"
database_name = "robocrystallographer"
dataSource = "Cluster0"
app_name = "data-cqjnk"
cluster_uri = "cluster0.fyeompa"

cpus_per_task, partition, account = [12, "lonepeak-guest", "owner-guest"]
# cpus_per_task, partition, account = [4, "notchpeak-freecycle", "sparks"]
# cpus_per_task, partition, account = [20, "notchpeak-guest", "owner-guest"]  # 32
# cpus_per_task, partition, account = [20, "ash-guest", "smithp-guest"]
# cpus_per_task, partition, account = [
#     2,
#     "notchpeak-shared-short",
#     "notchpeak-shared-short",
# ]  # need to specify amount of memory, kind of a "test" qos

# see also https://www.chpc.utah.edu/documentation/software/node-sharing.php
#########################

ray.shutdown()

with open("training/secrets.json", "r") as f:
    secrets = json.load(f)

# use requests directly to avoid pickle issues with submitit
MONGODB_API_KEY = secrets["MONGODB_API_KEY"]
username = secrets["PYMONGO_USERNAME"]
password = secrets["PYMONGO_PASSWORD"]

session_id = str(uuid4())

if dummy:
    tasks_structure = {
        "matbench_phonons": "last phdos peak",
        "matbench_perovskites": "e_form",
        # "matbench_mp_gap": "gap pbe",
        # "matbench_mp_e_form": "e_form",
    }
else:
    tasks_structure = {
        "matbench_phonons": "last phdos peak",
        "matbench_perovskites": "e_form",
        "matbench_mp_gap": "gap pbe",
        "matbench_mp_e_form": "e_form",
    }

# functions
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    out = []
    for i in range(0, len(lst), n):
        # yield lst[i:i + n]
        out.append(lst[i : i + n])
    return out


def robofeaturize(cifstr):
    """featurize a list of CIF strings using robycrys and assign ['failed'] for CIFs
    that produce an error"""
    featurizer = RobocrysFeaturizer({"use_conventional_cell": False, "symprec": 0.1})
    lbls = RobocrysFeaturizer().feature_labels()
    structure = Structure.from_str(cifstr, fmt="cif")
    features = featurizer.featurize(structure)
    # convert numpy types to python types
    features = [float(f) if isinstance(f, np.float64) else f for f in features]
    features = [int(f) if isinstance(f, np.int64) else f for f in features]
    results = {lbl: features[j] for j, lbl in enumerate(lbls)}
    return results


headers = {
    "Content-Type": "application/json",
    "Access-Control-Request-Headers": "*",
    "api-key": MONGODB_API_KEY,
}


class NpEncoder(json.JSONEncoder):
    """https://stackoverflow.com/a/57915246/13697228"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@ray.remote  # comment for debugging
def mongodb_robofeaturize(parameters: dict):
    """featurize a list of CIF strings using robycrys and assign ['failed'] for CIFs
    that produce an error"""
    cifstr = parameters["cif"]

    client = pymongo.MongoClient(
        f"mongodb+srv://{username}:{password}@{cluster_uri}.mongodb.net/?retryWrites=true&w=majority"  # noqa: E501
    )
    db = client[database_name]
    collection = db[collection_name]
    # results = collection.find_one({"cif": cifstr}) #  EXPENSIVE (e.g., $0.10/function call) without an index!
    # if results is None:
    t0 = time()
    try:
        results = robofeaturize(cifstr)
    except Exception as e:
        results = {"error": str(e)}
    print(results)
    utc = datetime.utcnow()
    results = {
        **parameters,
        **results,
        "dummy": dummy,
        "session_id": session_id,
        "timestamp": utc.timestamp(),
        "date": str(utc),
        "runtime": time() - t0,
    }
    collection.insert_one(results)
    return results


def mongodb_robofeaturize_batch(parameters_list):
    os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
    ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=cpus_per_task)
    return ray.get(
        [
            mongodb_robofeaturize.remote(parameters)  # comment for debugging
            # mongodb_robofeaturize(parameters)  # uncomment for debugging
            for parameters in parameters_list
        ]
    )


# SLURM/submitit commands
##setup
log_folder = "log_test/%j"
if dummy:
    walltime = 10
    chunksize = 5
else:
    walltime = int(np.round(120 / cpus_per_task)) + 5
    chunksize = 180

dfs = []
for key in tqdm(tasks_structure.keys()):
    with open(f"matbench_datasets/{key}_structures.pkl", "rb") as f:
        df = pickle.load(f)
        if dummy:
            df = df.sample(n=10)
        dfs.append(df)
        del df

df = pd.concat(dfs)
df.drop_duplicates(subset=["cif"], inplace=True)

# to find this string, click connect to your MongoDB cluster on the website
# also needed to go to "Network Access", click "Add IP address", click "Allow access
# from anywhere", and add
client = pymongo.MongoClient(
    f"mongodb+srv://{username}:{password}@{cluster_uri}.mongodb.net/?retryWrites=true&w=majority"  # noqa: E501
)
db = client[database_name]
collection = db[collection_name]

posts = collection.find({"cif": {"$exists": True}}, projection=["cif"])
cifs = [post["cif"] for post in tqdm(posts)]

mongo_df = pd.DataFrame(cifs, columns=["cif"])
df = df[~df.cif.isin(mongo_df.cif)]

records = df.to_dict("records")

# shuffle records
random.shuffle(records)

# # example record:
# {
#     "cif": "# generated using py...000000  1\n",
#     "target": 98.58577122703691,
#     "task": "matbench_phonons",
# }

# # uncomment for some debugging:
# robofeaturize(records[0]["cif"])
# mongodb_robofeaturize.remote(records[0])
# mongodb_robofeaturize(records[0])
# mongodb_robofeaturize_batch(records[:2])

# split data into chunks
cifstr_chunks = chunks(records, chunksize)

##execution
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_cpus_per_task=cpus_per_task,
    slurm_additional_parameters={"ntasks": 1, "account": account},
)

# sbatch array
jobs = executor.map_array(mongodb_robofeaturize_batch, cifstr_chunks)

# save jobs as pkl in case you want to look back at log output
with open(f"matbench_datasets/jobs/robofeat_jobs.pkl", "wb") as f:
    pickle.dump(jobs, f)

# comment this line if you want to submit all jobs at once, but you might run into
# limits on the number of jobs
results = [job.result() for job in jobs]

with open(f"matbench_datasets/jobs/robofeat_results.pkl", "wb") as f:
    pickle.dump(results, f)

1 + 1
# %% Code Graveyard

# def robofeaturize(structures):
#     """featurize a list of CIF strings using robycrys and assign ['failed'] for CIFs that produce an error"""
#     featurizer = RobocrysFeaturizer({"use_conventional_cell": False, "symprec": 0.1})
#     nids = len(structures)
#     features = [None] * nids
#     for i in range(nids):
#         structure = structures[i]
#         try:
#             features[i] = featurizer.featurize(structure)
#         except:
#             print("failed ID: " + str(i))
#             features[i] = ["failed"]
#     return features

# df3 = df3.drop(
#     df3[df3[lbls[0]] == "failed"].index
# )  # remove failed rows, lbls[0] is mineral_prototype as of 2021-05-27


# jobs_list = []

#    # jobs_list.append(jobs)

# num = random.randint(0, 1000)

# with open(f"matbench_datasets/jobs/robofeat_jobs_{num}.pkl", "wb") as f:
#     pickle.dump(jobs_list, f)

# with open(f"matbench_datasets/jobs/robofeat_jobs_{num}.pkl", "rb") as f:
#     jobs_list = pickle.load(f)

# for i, key in enumerate(tqdm(tasks_structure.keys())):
#     with open(f"matbench_datasets/{key}_structures.pkl", "rb") as f:
#         df = pickle.load(f)
#         df["structure"] = df["structure"].apply(lambda x: x.to(fmt="cif"))

#    # if dummy:
#    #     df = df.head(10)

#    # jobs = jobs_list[i]

# concatenation
# njobs = len(jobs)
# output = []
# for i in range(njobs):
#     output.append(jobs[i].result())

# # save features
# fname = f"matbench_datasets/{key}_robocrys_features"
# if dummy:
#     fname += "_dummy"
# with open(fname + ".pkl", "wb") as f:
#     pickle.dump(output, f)

# # combine MP properties and robocrys features
# lbls = RobocrysFeaturizer().feature_labels()
# df2 = df["target"]
# df = pd.DataFrame(
#     [i if isinstance(i, list) else [i] for j in output for i in j], columns=lbls
# )  # stack output from multiple jobs
# df3 = pd.concat([df2, df], axis=1)  # combine robocrys and MP props
# df3[df3[lbls[0]] == "failed"] = np.nan

# # save combined dataframe
# # df = pd.DataFrame( output, columns=lbls )
# fname = f"matbench_datasets/{key}_robocrys_features"
# if dummy:
#     fname += "_dummy"
# df3.to_csv(fname + ".csv")


# import random
# import numpy as np
# import pandas as pd
# import pymongo

# username = secrets["PYMONGO_USERNAME"]
# password = secrets["PYMONGO_PASSWORD"]

# to find this string, click connect to your MongoDB cluster
# client = pymongo.MongoClient(
#     "mongodb+srv://{username}:{password}@cluster0.fyeompa.mongodb.net/?retryWrites=true&w=majority"  # noqa: E501
# )
# db = client["robocrystallographer"]
# collection = db["matbench-datasets"]

# id = collection.insert_one(results).inserted_id  # type: ignore


# for key in tqdm(tasks_structure.keys()):

# df["task"] = key
# df.drop_duplicates(subset=["cif"], inplace=True)
# records = df.to_dict("records")


# @ray.remote  # comment for debugging
# def mongodb_robofeaturize(parameters: dict, verbose=True):
#     """featurize a list of CIF strings using robycrys and assign ['failed'] for CIFs
#     that produce an error"""
#     url_base = f"https://data.mongodb-api.com/app/{app_name}/endpoint/data/v1/action/"
#     insert_url = url_base + "insertOne"
#     find_url = url_base + "findOne"
#     cifstr = parameters["cif"]
#     payload = json.dumps(
#         {
#             "collection": collection_name,
#             "database": database_name,
#             "dataSource": dataSource,
#             "filter": {"cif": cifstr},
#         }
#     )
#     response = requests.request("POST", find_url, headers=headers, data=payload)
#     if response.status_code != 200:
#         raise ValueError(response.text)
#     results = response.json()["document"]

#     if results is None:
#         t0 = time()
#         try:
#             results = robofeaturize(cifstr)
#         except Exception as e:
#             results = {"error": str(e)}
#         print(results)
#         utc = datetime.utcnow()
#         results = {
#             **parameters,
#             **results,
#             "dummy": dummy,
#             "session_id": session_id,
#             "timestamp": utc.timestamp(),
#             "date": str(utc),
#             "runtime": time() - t0,
#         }
#         payload = json.dumps(
#             {
#                 "collection": collection_name,
#                 "database": database_name,
#                 "dataSource": dataSource,
#                 "document": results,
#             },
#             cls=NpEncoder,
#         )
#         if verbose:
#             print(f"Submitting {payload} to {insert_url}...")

#         response = requests.request("POST", insert_url, headers=headers, data=payload)
#     return results
