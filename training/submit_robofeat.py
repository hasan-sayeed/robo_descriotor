from time import time
from uuid import uuid4
import submitit
import pickle
from robocrys.featurize.featurizer import RobocrysFeaturizer
from tqdm import tqdm
from pymatgen.core import Structure
import datetime
import json
import requests

with open("training/secrets.json", "r") as f:
    secrets = json.load(f)

# use requests directly to avoid pickle issues with submitit
MONGODB_API_KEY = secrets["MONGODB_API_KEY"]

dummy = True

collection = "matbench-datasets"
database = "robocrystallographer"
dataSource = "Cluster0"
app_name = "data-cqjnk"

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
    try:
        features = featurizer.featurize(structure)
        results = {lbl: features[j] for j, lbl in enumerate(lbls)}
    except Exception as e:
        print(e)
        results = {"error": e}
    return results


headers = {
    "Content-Type": "application/json",
    "Access-Control-Request-Headers": "*",
    "api-key": MONGODB_API_KEY,
}


def mongodb_robofeaturize(parameters: dict, verbose=True):
    """featurize a list of CIF strings using robycrys and assign ['failed'] for CIFs
    that produce an error"""
    url_base = f"https://data.mongodb-api.com/app/{app_name}/endpoint/data/v1/action/"
    insert_url = url_base + "insertOne"
    find_url = url_base + "findOne"
    cifstr = parameters["cif"]
    # results = collection.find_one({"cifstr": cifstr})
    payload = json.dumps(
        {
            "collection": collection,
            "database": database,
            "dataSource": dataSource,
            "filter": {"cif": cifstr},
        }
    )
    response = requests.request("POST", find_url, headers=headers, data=payload)
    results = response.json()["document"]
    if results is not None:
        t0 = time()
        results = robofeaturize(cifstr)
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
        # id = collection.insert_one(results).inserted_id  # type: ignore
        payload = json.dumps(
            {
                "collection": collection,
                "database": database,
                "dataSource": dataSource,
                "document": results,
            }
        )
        if verbose:
            print(f"Submitting {payload} to {insert_url}...")

        response = requests.request("POST", insert_url, headers=headers, data=payload)
    return results


def mongodb_robofeaturize_batch(parameters_list):
    return [mongodb_robofeaturize(parameters) for parameters in parameters_list]


# SLURM/submitit commands
##setup
log_folder = "log_test/%j"
if dummy:
    walltime = 10
    chunksize = 5
else:
    walltime = 300
    chunksize = 360

for key in tqdm(tasks_structure.keys()):
    with open(f"matbench_datasets/{key}_structures.pkl", "rb") as f:
        df = pickle.load(f)

    if dummy:
        df = df.head(10)

    df["task"] = key
    records = df.to_dict("records")

    # mongodb_robofeaturize(records[0])

    # e.g.,
    # {
    #     "cif": "# generated using py...000000  1\n",
    #     "target": 98.58577122703691,
    #     "task": "matbench_phonons",
    # }

    # split data into chunks
    cifstr_chunks = chunks(records, chunksize)

    # partition, account = ["notchpeak-freecycle", "sparks"]
    partition, account = ["notchpeak-guest", "owner-guest"]

    ##execution
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=walltime,
        slurm_partition=partition,
        slurm_cpus_per_task=1,
        slurm_additional_parameters={"ntasks": 1, "account": account},
    )

    # sbatch array
    jobs = executor.map_array(mongodb_robofeaturize_batch, cifstr_chunks)

    # save jobs as pkl in case you want to look back at log output
    with open(f"matbench_datasets/jobs/{key}_robofeat_jobs.pkl", "wb") as f:
        pickle.dump(jobs, f)

    # comment this line if you want to submit all jobs at once, but you might run into
    # limits on the number of jobs
    results = [job.result() for job in jobs]

    with open(f"matbench_datasets/jobs/{key}_robofeat_results.pkl", "wb") as f:
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
