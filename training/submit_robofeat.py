import numpy as np
import submitit
import pickle
import pandas as pd
from robocrys.featurize.featurizer import RobocrysFeaturizer
from tqdm import tqdm
from pymatgen.core import Structure

dummy = True

if dummy:
    tasks_structure = {
        "matbench_phonons": "last phdos peak",
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


def robofeaturize(cifstr_list):
    """featurize a list of CIF strings using robycrys and assign ['failed'] for CIFs that produce an error"""
    featurizer = RobocrysFeaturizer({"use_conventional_cell": False, "symprec": 0.1})
    nids = len(cifstr_list)
    features = [None] * nids
    for i in range(nids):
        cifstr = cifstr_list[i]
        structure = Structure.from_str(cifstr, fmt="cif")
        try:
            features[i] = featurizer.featurize(structure)
        except Exception as e:
            print(f"failed ID: {str(i)}, {e}")
            features[i] = [e]
    return features


for key in tqdm(tasks_structure.keys()):
    with open(f"matbench_datasets/{key}_structures.pkl", "rb") as f:
        df = pickle.load(f)
        df["structure"] = df["structure"].apply(lambda x: x.to(fmt="cif"))

    if dummy:
        df = df.head(10)

    # SLURM/submitit commands
    ##setup
    log_folder = "log_test/%j"
    walltime = 240  # 300
    chunksize = 260  # 360
    pars = chunks(df.structure.tolist(), chunksize)  # split CIFs into chunks
    # partition, account = ['lonepeak','sparks']
    partition, account = ["lonepeak-guest", "owner-guest"]

    ##execution
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=walltime,
        slurm_partition=partition,
        slurm_additional_parameters={"account": account},
    )

    jobs = executor.map_array(robofeaturize, pars)  # sbatch array

    # concatenation
    njobs = len(jobs)
    output = []
    for i in range(njobs):
        output.append(jobs[i].result())

    # save features
    with open(f"matbench_datasets/{key}_robocrys_features.pkl", "wb") as f:
        pickle.dump(output, f)

    # combine MP properties and robocrys features
    lbls = RobocrysFeaturizer().feature_labels()
    df2 = df["target"]
    df = pd.DataFrame(
        [i if isinstance(i, list) else [i] for j in output for i in j], columns=lbls
    )  # stack output from multiple jobs
    df3 = pd.concat([df2, df], axis=1)  # combine robocrys and MP props
    df3[df3[lbls[0]] == "failed"] = np.nan

    # save combined dataframe
    # df = pd.DataFrame( output, columns=lbls )
    df3.to_csv(f"matbench_datasets/{key}_robocrys_features.csv")

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
