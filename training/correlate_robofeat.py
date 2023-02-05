import pickle
import pandas as pd

with open("matbench_datasets/robofeat.pkl", "rb") as f:
    feat_df = pickle.load(f)

with open("matbench_datasets/matbench_mp_e_form_structures.pkl", "rb") as f:
    mp_e_form_df = pickle.load(f)

with open("matbench_datasets/matbench_mp_gap_structures.pkl", "rb") as f:
    mp_gap_df = pickle.load(f)

with open("matbench_datasets/matbench_perovskites_structures.pkl", "rb") as f:
    perovskites_df = pickle.load(f)

with open("matbench_datasets/matbench_phonons_structures.pkl", "rb") as f:
    phonons_df = pickle.load(f)

feat_df.drop_duplicates(subset=["cif"], inplace=True)


def find_matching_index(list1, list2):
    """https://stackoverflow.com/a/49247599/13697228"""

    # Create an inverse index which keys are now sets
    inverse_index = {}

    for index, element in enumerate(list1):

        if element not in inverse_index:
            inverse_index[element] = {index}

        else:
            inverse_index[element].add(index)

    # Traverse the second list
    matching_index = []

    for index, element in enumerate(list2):

        # We have to create one pair by element in the set of the inverse index
        if element in inverse_index:
            matching_index.extend([(x, index) for x in inverse_index[element]])

    return matching_index


def lookup_features(feat_df, prop_df):
    matches = find_matching_index(feat_df["cif"].tolist(), prop_df["cif"].tolist())

    left_idx, right_idx = zip(*matches)
    left_idx = list(left_idx)
    right_idx = list(right_idx)

    mp_e_form_feat = (
        feat_df.drop(
            columns=[
                "_id",
                "target",
                "task",
                "dummy",
                "session_id",
                "timestamp",
                "date",
                "runtime",
                "error",
            ]
        )
        .iloc[left_idx]
        .reset_index(drop=True)
    )

    mp_e_form_targ = prop_df["target"].iloc[right_idx].reset_index(drop=True)

    return pd.concat([mp_e_form_feat, mp_e_form_targ], axis=1)


mp_e_form_feat_df = lookup_features(feat_df, mp_e_form_df)
mp_gap_feat_df = lookup_features(feat_df, mp_gap_df)
perovskites_feat_df = lookup_features(feat_df, perovskites_df)
phonons_feat_df = lookup_features(feat_df, phonons_df)

mp_e_form_feat_df.to_csv("matbench_datasets/mp_e_form_feat.csv", index=False)
mp_gap_feat_df.to_csv("matbench_datasets/mp_gap_feat.csv", index=False)
perovskites_feat_df.to_csv("matbench_datasets/perovskites_feat.csv", index=False)
phonons_feat_df.to_csv("matbench_datasets/phonons_feat.csv", index=False)

1 + 1

# %%
# mapper = feat_df.set_index("cif").drop(
#     columns=[
#         "_id",
#         "target",
#         "task",
#         "dummy",
#         "session_id",
#         "timestamp",
#         "date",
#         "runtime",
#         "error",
#     ]
# )

# mapper.head(2).to_dict(orient="index")

# # replace
# mp_e_form_df.head(2).replace(mapper)
