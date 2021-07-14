from functools import reduce
from typing import List

from snakemake.io import glob_wildcards

import pandas as pd
import numpy as np

import os

N_JOBS, MAX_ITER, MAX_NR = 28, 100, 20

MODEL_NAMES = [
    # "lda", "bayes",
    # "log_reg",
    "rf"
]
META_MODEL_NAMES = [
    "stacking",
    "voting_hard",
    "voting_soft"
]


def get_csv_names(dataset):
    wc = glob_wildcards(f"data/{dataset}/csv/all/{{csv_names}}")[0]
    return [e for e in wc if "csv" in e]


def get_model():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    return {
        "lda": LinearDiscriminantAnalysis(),
        "bayes": GaussianNB(),
        "log_reg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_jobs=-1)
    }


def get_meta_model():
    from optimizer.ensemble import StackingClassifier
    from optimizer.ensemble import VotingClassifier
    return {
        "stacking": StackingClassifier(estimators=None, n_jobs=-1),
        "voting_hard": VotingClassifier(estimators=None, voting="hard", n_jobs=-1),
        "voting_soft": VotingClassifier(estimators=None, voting="soft", n_jobs=-1)
    }


def concat_datasets(paths_list):
    encoded_datasets_tmp = [pd.read_csv(p, index_col=0) for p in paths_list]
    df_dummy = get_all_present_indices_df(encoded_datasets_tmp)
    df = pd.concat([df.loc[df_dummy.index, :].iloc[:, :-1].sort_index()
                    for df in encoded_datasets_tmp], axis=1)
    return df.values, df_dummy.y.values


def get_all_present_indices_df(df_list: List[pd.DataFrame]):
    # get indices present in all encoded datasets: {1,2,3,4,5}, {1,3,5}, {1,2,3,4} -> {1,3}
    idcs_new = sorted(
        set.intersection(*[set(df.index) for df in df_list]),
        key=lambda n: int(n.split("_")[1])
    )
    df_dummy = pd.DataFrame(np.zeros((len(idcs_new), 1)), index=idcs_new)
    df_dummy["y"] = df_list[0].loc[idcs_new, "y"]
    return df_dummy
