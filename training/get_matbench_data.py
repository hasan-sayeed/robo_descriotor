import pickle
from matminer.datasets import load_dataset
from pymatgen.core.composition import Composition
from tqdm import tqdm

# Download datasets from Matbench that's input is 'structure'.

tasks_structure = {
    "matbench_phonons": "last phdos peak",
    "matbench_perovskites": "e_form",
    "matbench_mp_gap": "gap pbe",
    "matbench_mp_e_form": "e_form",
}

for index, (key, value) in enumerate(tqdm(tasks_structure.items())):
    df = load_dataset(key, pbar=True)

    formula = []
    for s in df.structure:
        formula.append(s.formula)

    df["formula"] = formula
    # del df['structure']
    df = df.rename({value: "target"}, axis=1)
    df[["formula", "target"]].to_csv(f"matbench_datasets/{key}.csv", index=False)
    with open(f"matbench_datasets/{key}_structures.pkl", "wb") as f:
        pickle.dump(df[["structure", "target"]], f)


# Download datasets from Matbench that's input is 'composition'.

tasks_composition = {
    "matbench_steels": "yield strength",
    "matbench_expt_gap": "gap expt",
    "matbench_expt_is_metal": "is_metal",
    "matbench_glass": "gfa",
}

for index, (key, value) in enumerate(tasks_composition.items()):
    df = load_dataset(key)

    formula = []
    for s in df.composition:
        formula.append(Composition(s).formula)

    df["formula"] = formula
    del df["composition"]
    df = df.rename({value: "target"}, axis=1)
    df = df[["formula", "target"]]
    df.to_csv("matbench_datasets/{}.csv".format(key), index=False)

1+1

# %% Code Graveyard

# tasks_structure = {'matbench_jdft2d': 'exfoliation_en', 'matbench_phonons': 'last phdos peak', 'matbench_dielectric': 'n', 'matbench_log_gvrh': 'log10(G_VRH)',
#          'matbench_log_kvrh': 'log10(K_VRH)', 'matbench_perovskites': 'e_form', 'matbench_mp_gap': 'gap pbe', 'matbench_mp_is_metal': 'is_metal',
#          'matbench_mp_e_form': 'e_form'}
