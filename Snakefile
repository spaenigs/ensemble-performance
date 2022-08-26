import itertools
import os.path

import pandas as pd
import numpy as np
import altair as alt

import functools
import math

import yaml
from functools import reduce
from numpy.random import default_rng
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit # Monte Carlo CV
from sklearn.tree import DecisionTreeClassifier
from more_itertools import chunked

from utils import get_csv_names
from optimizer.ensemble import StackingClassifier, VotingClassifier

from optimizer.optimization import BinaryMVO
import optimizer.optimization.fitness_function as ff

FOLDS = range(6)#100)

MODEL = {
    # "lr": LogisticRegression(max_iter=1000),
    # "dt": DecisionTreeClassifier(),
    # "bayes": GaussianNB(),
    # "rf":  RandomForestClassifier(),
    "mlp": MLPClassifier(max_iter=1000)
}
MODELS = list(MODEL.keys())

META_MODEL = {
    "stacking": StackingClassifier(estimators=None),
    # "voting_soft": VotingClassifier(estimators=None, voting="soft"),
    # "voting_hard": VotingClassifier(estimators=None, voting="hard")
}
META_MODELS = list(META_MODEL.keys())

DATASETS = [
    "avp_amppred",
    # "amp_antibp2",
    # "isp_il10pred",
    # "cpp_mlcpp-complete",
    # "nep_neuropipred-complete",
    # "pip_pipel",
    # "aip_antiinflam-complete",
    # "acp_mlacp",
    # "atb_antitbp",
    # "hem_hemopi"
]

N_ENCODINGS = None

wildcard_constraints:
    fold="\d+"

rule all:
    input:
        expand("data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv",
            dataset=DATASETS, model=MODELS,
               fold=FOLDS),

        expand("data/temp/{dataset}/ensembles_res/{meta_model}/{model}/res.csv",
               dataset=DATASETS, meta_model=META_MODELS, model=MODELS),

        # expand("data/temp/{dataset}/ensemble_mvo/{meta_model}/{model}/gens_vs_perf_{fold}.txt",
        #        dataset=DATASETS, meta_model=META_MODELS, model=MODELS,
        #        fold=[
        #            0,
        #            1, 2, 3, 4
        #        ]
        #
        # ),
        # expand("data/temp/{dataset}/vis/gen_vs_perf.html", dataset=DATASETS),
        expand("data/temp/{dataset}/single_encodings/{model}/res.csv",
               model=MODELS, dataset=DATASETS),
        # expand("data/temp/{dataset}/areas/{model}/res.csv",
        #        dataset=DATASETS, model=MODELS),
        # expand("data/temp/{dataset}/ensembles_res/res.csv",
        #        dataset=DATASETS),
        #
        # expand("data/temp/{dataset}/ensembles_res/cd.yaml", dataset=DATASETS),
        # expand("data/temp/{dataset}/vis/kappa_error_plot.html", dataset=DATASETS),
        # expand("data/temp/{dataset}/vis/box_plot.html", dataset=DATASETS),
        # expand("data/temp/{dataset}/vis/xcd_plot.html", dataset=DATASETS),
        # expand("data/temp/{dataset}/vis/box_plot_manova.html", dataset=DATASETS),
        # expand("data/temp/{dataset}/stats/table.html", dataset=DATASETS),
        # expand("data/temp/{dataset}/{dataset}_zipped.zip", dataset=DATASETS),
        # "data/temp/all_datasets/tables/dataset_tables.html",
        # "data/temp/all_datasets/tables/areas_table.html"


# search for common indices across all datasets (less indices due to sec + ter struc.)
rule common_idx:
    input:
        lambda wildcards:
            expand(f"data/{wildcards.dataset}/csv/all/{{csv_name}}",
                   csv_name=get_csv_names(wildcards.dataset)[:N_ENCODINGS])
    output:
        "data/temp/{dataset}/indices.csv"
    run:
        dict_indcs = {}
        indcs = []
        for p in list(input):
            df = pd.read_csv(p, index_col=0)
            dict_indcs = dict_indcs | dict(df["y"])
            indcs += [list(df.index)]

        df_res = pd.DataFrame(dict_indcs.items(), columns=["idx", "y"])

        # get common all indices
        indcs = sorted(functools.reduce(set.intersection, indcs[1:], set(indcs[0])))
        df_res.loc[df_res.idx.isin(indcs)].to_csv(output[0])

# compute indices per fold
rule splits:
    input:
        "data/temp/{dataset}/indices.csv",
    output:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv"
    run:
        df_indices = pd.read_csv(input[0], index_col=0)
        indices, y = df_indices["idx"].values, df_indices.y.values

        gss = StratifiedShuffleSplit(n_splits=len(FOLDS), train_size=.8, random_state=42)

        df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for train_idx, test_idx in gss.split(indices, y):
            val_indcs = test_idx[:math.ceil(len(test_idx) / 2)]
            test_indcs = test_idx[math.ceil(len(test_idx) / 2):]
            ser_indices_train = df_indices.iloc[train_idx, :]["idx"]
            df_train = pd.concat([
                df_train,
                ser_indices_train.reset_index(drop=True)
            ], axis=1)
            ser_indices_val = df_indices.iloc[val_indcs, :]["idx"]
            df_val = pd.concat([
                df_val,
                ser_indices_val.reset_index(drop=True)
            ],axis=1)
            ser_indices_test = df_indices.iloc[test_indcs, :]["idx"]
            df_test = pd.concat([
                df_test,
                ser_indices_test.reset_index(drop=True)
            ], axis=1)

        df_train.columns = [f"fold_{i}" for i in FOLDS]
        df_val.columns = [f"fold_{i}" for i in FOLDS]
        df_test.columns = [f"fold_{i}" for i in FOLDS]

        df_train.to_csv(output[0])
        df_val.to_csv(output[1])
        df_test.to_csv(output[2])

# remove indices, scale dataset, remove zero variance columns
rule scale:
    input:
        "data/temp/{dataset}/indices.csv",
        "data/{dataset}/csv/all/{csv_name}"
    output:
        "data/temp/{dataset}/csv/scaled/{csv_name}"
    run:
        df_indices = pd.read_csv(input[0], index_col=0)
        indices = df_indices["idx"].values

        df = pd.read_csv(input[1], index_col=0).loc[indices, ]
        X, y = df.iloc[:, :-1].values, df["y"].values
        X_scaled = MinMaxScaler().fit_transform(X)

        vals = np.hstack((X_scaled, y.reshape((y.shape[0], 1))))

        indices = np.argwhere(pd.DataFrame(vals).std().values == 0).flatten()
        vals = np.delete(vals, indices, 1)

        if len(indices) != 0:
            print(wildcards.csv_name)

        df_res = pd.DataFrame(vals, columns=np.delete(df.columns, indices), index=df.index)
        df_res.to_csv(output[0])

rule single_encodings:
    input:
        "data/temp/{dataset}/csv/scaled/{csv_name}",
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv"
    output:
        "data/temp/{dataset}/single_encodings/{model}/{fold}/{csv_name}"
    run:
        df_indcs_train = pd.read_csv(input[1], index_col=0)
        indcs_train_tmp = df_indcs_train[f"fold_{wildcards.fold}"]
        df_indcs_val = pd.read_csv(input[2],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]
        indcs_train = pd.concat([indcs_train_tmp, indcs_val])

        df_indcs_test = pd.read_csv(input[3], index_col=0)
        indcs_test = df_indcs_test[f"fold_{wildcards.fold}"]

        df = pd.read_csv(input[0], index_col=0)
        X_train = df.iloc[:, :-1].loc[indcs_train, :].values
        y_train = df.loc[indcs_train, "y"].values
        X_test = df.iloc[:, :-1].loc[indcs_test, :].values
        y_test = df.loc[indcs_test, "y"].values

        clf = MODEL[wildcards.model]
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mcc = matthews_corrcoef(y_test, y_pred)
        except np.linalg.LinAlgError as e:
            print(e)
            mcc = 0.0
        except ValueError as e:
            print(e)
            mcc = 0.0

        pd.DataFrame({
            "mcc": [mcc],
            "fold": [wildcards.fold],
            "encoding": [wildcards.csv_name],
            "model": [wildcards.model]
        }).to_csv(output[0])

rule collect_single_encodings:
    input:
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/single_encodings/{wildcards.model}/{{fold}}/{{csv_name}}",
                   fold=FOLDS, csv_name=get_csv_names(wildcards.dataset)[:N_ENCODINGS])
    output:
        "data/temp/{dataset}/single_encodings/{model}/res.csv"
    run:
        df_res = pd.DataFrame()
        for p in list(input):
            df_tmp = pd.read_csv(p, index_col=0)
            df_res = pd.concat([df_res, df_tmp])

        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost',port=8888,stdoutToServer=True,stderrToServer=True)

        for csv_name in get_csv_names(wildcards.dataset):
            print(df_res.groupby("encoding").apply(lambda df: df.mcc.mean()))

        df_res.to_csv(output[0])


def kappa(y_pred_1, y_pred_2):
    a, b, c, d = 0, 0, 0, 0
    for i, j in zip(y_pred_1, y_pred_2):
        if (i, j) == (1, 1):
            a += 1
        elif (i, j) == (0, 0):
            d += 1
        elif (i, j) == (1, 0):
            b += 1
        elif (i, j) == (0, 1):
            c += 1
    a, b, c, d = \
        [v/len(y_pred_1) for v in [a, b, c, d]]
    dividend = 2 * (a*d-b*c)
    divisor = ((a+b) * (b+d)) + ((a+c) * (c+d))
    try:
        return dividend / divisor
    except ZeroDivisionError:
        return 0.0

rule kappa_error_all:
    input:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/csv/scaled/{{csv_name}}",
                   csv_name=get_csv_names(wildcards.dataset)[:N_ENCODINGS])
    output:
        "data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv"
    threads:
        8
    run:
        df_indcs_train = pd.read_csv(input[0], index_col=0)
        indcs_train = df_indcs_train[f"fold_{wildcards.fold}"]

        df_indcs_val = pd.read_csv(input[1],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]

        paths = list(input[2:])

        encoded_datasets = [pd.read_csv(p, index_col=0) for p in paths]

        X_train_list = \
            [df.loc[indcs_train, :].iloc[:, :-1].values
             for df in encoded_datasets]

        X_val_list = \
            [df.loc[indcs_val, :].iloc[:, :-1].values
             for df in encoded_datasets]


        y_train, y_val = \
            encoded_datasets[0].loc[indcs_train, "y"].values, \
            encoded_datasets[0].loc[indcs_val, "y"].values

        clf = MODEL[wildcards.model]
        eclf = META_MODEL[META_MODELS[0]]
        eclf.estimators = [(paths[i], clf) for i in range(len(paths))]
        eclf.fit(X_train_list, y_train)

        res = []
        for ((e1, clf_1), X_val_1), ((e2, clf_2), X_val_2) in \
                itertools.combinations(zip(eclf.estimators_, X_val_list), 2):
            y_pred_tree_1, y_pred_tree_2 = \
                clf_1.predict(X_val_1), clf_2.predict(X_val_2)
            error_1, error_2 = \
                1 - accuracy_score(y_pred_tree_1, y_val), \
                1 - accuracy_score(y_pred_tree_2, y_val)
            mean_pairwise_error = np.mean([error_1, error_2])
            k = kappa(y_pred_tree_1,y_pred_tree_2)
            res += [[k, mean_pairwise_error, e1, e2]]

        df_res = pd.DataFrame(res, columns=["x", "y", "encoding_1", "encoding_2"])
        df_res["model"] = wildcards.model

        df_res.to_csv(output[0])

rule chull_complete:
    input:
        "data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/chull_complete/{model}/{fold}.csv",
        "data/temp/{dataset}/chull_complete/{model}/area_{fold}.csv"
    run:
        df_points = pd.read_csv(input[0], index_col=0)

        hull = ConvexHull(df_points[["x", "y"]])

        df_points["chull_complete"] = -1
        df_points.iloc[hull.vertices, df_points.columns.get_loc("chull_complete")] = \
            range(hull.vertices.shape[0])

        df_points.to_csv(output[0])

        pd.DataFrame({
            "model": [wildcards.model],
            "area": [hull.area],
            "fold": [wildcards.fold]
        }).to_csv(output[1])

rule collect_areas:
    input:
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/chull_complete/{wildcards.model}/area_{{fold}}.csv",
                   fold=FOLDS)
    output:
        "data/temp/{dataset}/areas/{model}/res.csv"
    run:
        df_res = pd.DataFrame()
        for p in list(input):
            df_res = pd.concat([df_res, pd.read_csv(p, index_col=0)])

        df_res.to_csv(output[0])

# copied from Kuncheva, L.: Combining Pattern Classifers (Wiley) p. 288
def pareto_n(a):
    # N --> rows
    # n --> cols
    N, n = a.shape
    Mask = np.zeros((N,))
    # mask the first point
    # Mask[0] = 1
    # iterate over each remaining point
    for i in range(N):
        flag = 0
        # amount of masked points, i.e., not in pareto frontier
        SM = sum(Mask)
        # get indices of masked points
        P = np.nonzero(Mask)[0]
        # iter over amount of masked points
        for j in range(int(SM)):
            # a[i, :] --> one point in the cloud
            # P[j]    --> index of j-th masked point
            if np.sum(a[i, :] <= a[P[j], :]) == n:
                flag = 1
        if flag == 0:
            for j in range(int(SM)):
                if np.sum(a[P[j], :] <= a[i, :]) == n:
                    Mask[P[j]] = 0
            Mask[i] = 1
    return np.nonzero(Mask)

rule chull:
    input:
        "data/temp/{dataset}/chull_complete/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/chull/{model}/{fold}.csv"
    run:
        df_points = pd.read_csv(input[0], index_col=0)

        df_hull = df_points.loc[df_points.chull_complete != -1, ["x", "y"]]

        # mask convex hull (use only vals towards lower, left corner)
        P = pareto_n(-df_hull.values)

        indices = list(df_hull.iloc[P[0], :].sort_values("x").index)

        df_points["chull"] = -1
        df_points.iloc[indices, df_points.columns.get_loc("chull")] = range(len(indices))

        df_points.to_csv(output[0])

rule pfront:
    input:
        "data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/pfront/{model}/{fold}.csv"
    run:
        df_points = pd.read_csv(input[0], index_col=0)

        P = pareto_n(-df_points[["x", "y"]].values)

        indices = list(df_points.iloc[P[0], :].sort_values("x").index)

        df_points["pfront"] = -1
        df_points.iloc[indices, df_points.columns.get_loc("pfront")] = range(len(indices))

        df_points.to_csv(output[0])

rule ensemble_bst:
    input:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv",
        "data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/{fold}.csv",
        "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/kappa_error_{fold}.csv"
    run:
        df_indcs_train = pd.read_csv(input[0],index_col=0)
        indcs_train_tmp = df_indcs_train[f"fold_{wildcards.fold}"]
        df_indcs_val = pd.read_csv(input[1],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]
        indcs_train = pd.concat([indcs_train_tmp, indcs_val])

        df_indcs_test = pd.read_csv(input[2],index_col=0)
        indcs_test = df_indcs_test[f"fold_{wildcards.fold}"]

        df_points = pd.read_csv(input[3], index_col=0)

        # y is average pairwise error
        train_paths = list(set(
            df_points\
                .sort_values("y").iloc[:15, :][["encoding_1", "encoding_2"]]\
                .values.flatten()
        ))

        # keep ensemble best encodings position for later usage
        indices = df_points.sort_values("y").iloc[:15, :].index
        df_points["ensemble_best"] = False
        df_points.iloc[indices, df_points.columns.get_loc("ensemble_best")] = True

        encoded_datasets = [pd.read_csv(p, index_col=0) for p in train_paths]

        X_train_list, X_test_list = \
            [df.loc[indcs_train, :].iloc[:, :-1].values
             for df in encoded_datasets], \
            [df.loc[indcs_test, :].iloc[:, :-1].values
             for df in encoded_datasets]

        y_train, y_test = \
            encoded_datasets[0].loc[indcs_train, "y"].values, \
            encoded_datasets[0].loc[indcs_test, "y"].values

        clf = MODEL[wildcards.model]
        eclf = META_MODEL[wildcards.meta_model]
        eclf.estimators = [(train_paths[i], clf) for i in range(len(train_paths))]

        try:
            eclf.fit(X_train_list, y_train)
            y_pred = eclf.predict(X_test_list)
            mcc = matthews_corrcoef(y_test,y_pred)
        except np.linalg.LinAlgError as e:
            print(e)
        except ValueError as e:
            print(e)

        pd.DataFrame({
            "mcc": [mcc],
            "fold": [wildcards.fold],
            "model": [wildcards.model],
            "meta_model": [wildcards.meta_model]
        }).to_csv(output[0])

        df_points.to_csv(output[1])

rule get_random_encodings:
    input:
        "data/temp/{dataset}/kappa_error_all/{model}/0.csv"
    output:
        "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/rand_encodings.csv"
    run:
        df_points = pd.read_csv(input[0], index_col=0)

        idcs = default_rng().choice(df_points.index, size=15, replace=False)

        pd.DataFrame(idcs, columns=["enc_index"]).to_csv(output[0])

rule ensemble_rnd:
    input:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv",
        "data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv",
        "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/rand_encodings.csv"
    output:
        "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/{fold}.csv",
        "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/kappa_error_{fold}.csv"
    run:
        df_indcs_train = pd.read_csv(input[0], index_col=0)
        indcs_train_tmp = df_indcs_train[f"fold_{wildcards.fold}"]
        df_indcs_val = pd.read_csv(input[1],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]
        indcs_train = pd.concat([indcs_train_tmp, indcs_val])

        df_indcs_test = pd.read_csv(input[2], index_col=0)
        indcs_test = df_indcs_test[f"fold_{wildcards.fold}"]

        df_points = pd.read_csv(input[3], index_col=0)

        idcs = pd.read_csv(input[4], index_col=0)["enc_index"]
        train_paths = list(set(
            df_points.iloc[idcs, :][["encoding_1", "encoding_2"]].values.flatten()
        ))

        # keep ensemble best encodings position for later usage
        df_points["ensemble_rand"] = False
        df_points.iloc[idcs, df_points.columns.get_loc("ensemble_rand")] = True

        encoded_datasets = [pd.read_csv(p, index_col=0) for p in train_paths]

        X_train_list, X_test_list = \
            [df.loc[indcs_train, :].iloc[:, :-1].values
             for df in encoded_datasets], \
            [df.loc[indcs_test, :].iloc[:, :-1].values
             for df in encoded_datasets]

        y_train, y_test = \
            encoded_datasets[0].loc[indcs_train, "y"].values, \
            encoded_datasets[0].loc[indcs_test, "y"].values

        clf = MODEL[wildcards.model]
        eclf = META_MODEL[wildcards.meta_model]
        eclf.estimators = [(train_paths[i], clf) for i in range(len(train_paths))]

        try:
            eclf.fit(X_train_list,y_train)
            y_pred = eclf.predict(X_test_list)
            mcc = matthews_corrcoef(y_test,y_pred)
        except np.linalg.LinAlgError as e:
            print(e)
        except ValueError as e:
            print(e)

        pd.DataFrame({
            "mcc": [mcc],
            "fold": [wildcards.fold],
            "model": [wildcards.model],
            "meta_model": [wildcards.meta_model]
        }).to_csv(output[0])

        df_points.to_csv(output[1])

rule ensemble_chull:
    input:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv",
        "data/temp/{dataset}/chull/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/ensemble_chull/{meta_model}/{model}/{fold}.csv",
        "data/temp/{dataset}/ensemble_chull/{meta_model}/{model}/kappa_error_{fold}.csv"
    run:
        df_indcs_train = pd.read_csv(input[0], index_col=0)
        indcs_train_tmp = df_indcs_train[f"fold_{wildcards.fold}"]
        df_indcs_val = pd.read_csv(input[1],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]
        indcs_train = pd.concat([indcs_train_tmp, indcs_val])

        df_indcs_test = pd.read_csv(input[2], index_col=0)
        indcs_test = df_indcs_test[f"fold_{wildcards.fold}"]

        df_points = pd.read_csv(input[3], index_col=0)

        train_paths = list(set(
            df_points.loc[df_points.chull != -1][["encoding_1", "encoding_2"]]\
                .values.flatten()
        ))

        # keep ensemble best encodings position for later usage
        indices = df_points.loc[df_points.chull != -1].index
        df_points["ensemble_chull"] = False
        df_points.iloc[indices, df_points.columns.get_loc("ensemble_chull")] = True

        encoded_datasets = [pd.read_csv(p, index_col=0) for p in train_paths]

        X_train_list, X_test_list = \
            [df.loc[indcs_train, :].iloc[:, :-1].values
             for df in encoded_datasets], \
            [df.loc[indcs_test, :].iloc[:, :-1].values
             for df in encoded_datasets]

        y_train, y_test = \
            encoded_datasets[0].loc[indcs_train, "y"].values, \
            encoded_datasets[0].loc[indcs_test, "y"].values

        clf = MODEL[wildcards.model]
        eclf = META_MODEL[wildcards.meta_model]
        eclf.estimators = [(train_paths[i], clf) for i in range(len(train_paths))]

        try:
            eclf.fit(X_train_list,y_train)
            y_pred = eclf.predict(X_test_list)
            mcc = matthews_corrcoef(y_test,y_pred)
        except np.linalg.LinAlgError as e:
            print(e)
        except ValueError as e:
            print(e)

        pd.DataFrame({
            "mcc": [mcc],
            "fold": [wildcards.fold],
            "model": [wildcards.model],
            "meta_model": [wildcards.meta_model]
        }).to_csv(output[0])

        df_points.to_csv(output[1])

rule ensemble_pfront:
    input:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv",
        "data/temp/{dataset}/pfront/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/ensemble_pfront/{meta_model}/{model}/{fold}.csv",
        "data/temp/{dataset}/ensemble_pfront/{meta_model}/{model}/kappa_error_{fold}.csv"
    run:
        df_indcs_train = pd.read_csv(input[0], index_col=0)
        indcs_train_tmp = df_indcs_train[f"fold_{wildcards.fold}"]
        df_indcs_val = pd.read_csv(input[1],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]
        indcs_train = pd.concat([indcs_train_tmp, indcs_val])

        df_indcs_test = pd.read_csv(input[2], index_col=0)
        indcs_test = df_indcs_test[f"fold_{wildcards.fold}"]

        df_points = pd.read_csv(input[3], index_col=0)

        train_paths = list(set(
            df_points.loc[df_points.pfront != -1][["encoding_1", "encoding_2"]] \
                .values.flatten()
        ))

        # keep ensemble best encodings position for later usage
        indices = df_points.loc[df_points.pfront != -1].index
        df_points["ensemble_pfront"] = False
        df_points.iloc[indices, df_points.columns.get_loc("ensemble_pfront")] = True

        encoded_datasets = [pd.read_csv(p, index_col=0) for p in train_paths]

        X_train_list, X_test_list = \
            [df.loc[indcs_train, :].iloc[:, :-1].values
             for df in encoded_datasets], \
            [df.loc[indcs_test, :].iloc[:, :-1].values
             for df in encoded_datasets]

        y_train, y_test = \
            encoded_datasets[0].loc[indcs_train, "y"].values, \
            encoded_datasets[0].loc[indcs_test, "y"].values

        clf = MODEL[wildcards.model]
        eclf = META_MODEL[wildcards.meta_model]
        eclf.estimators = [(train_paths[i], clf) for i in range(len(train_paths))]

        try:
            eclf.fit(X_train_list,y_train)
            y_pred = eclf.predict(X_test_list)
            mcc = matthews_corrcoef(y_test,y_pred)
        except np.linalg.LinAlgError as e:
            print(e)
        except ValueError as e:
            print(e)

        pd.DataFrame({
            "mcc": [mcc],
            "fold": [wildcards.fold],
            "model": [wildcards.model],
            "meta_model": [wildcards.meta_model]
        }).to_csv(output[0])

        df_points.to_csv(output[1])

rule ensemble_mvo:
    input:
        "data/temp/{dataset}/mcv_folds_train.csv",
        "data/temp/{dataset}/mcv_folds_val.csv",
        "data/temp/{dataset}/mcv_folds_test.csv",
        "data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv"
    output:
        "data/temp/{dataset}/ensemble_mvo/{meta_model}/{model}/{fold}.csv",
        "data/temp/{dataset}/ensemble_mvo/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_mvo/{meta_model}/{model}/gens_vs_perf_{fold}.txt"
    threads:
        1000
    run:
        # use complete for MVO inner cv
        df_indcs_train = pd.read_csv(input[0], index_col=0)
        indcs_train_tmp = df_indcs_train[f"fold_{wildcards.fold}"]
        df_indcs_val = pd.read_csv(input[1],index_col=0)
        indcs_val = df_indcs_val[f"fold_{wildcards.fold}"]
        indcs_train = pd.concat([indcs_train_tmp, indcs_val])

        # use for testing after optimization
        df_indcs_test = pd.read_csv(input[2], index_col=0)
        indcs_test = df_indcs_test[f"fold_{wildcards.fold}"]

        df_points = pd.read_csv(input[2], index_col=0)

        # y is average pairwise error
        train_paths = list(set(
            df_points[["encoding_1", "encoding_2"]] \
                .values.flatten()
        ))

        n_universes = 2 # 32
        max_generations = 15

        p_0 = 6 / len(train_paths)
        mvo = BinaryMVO(
            n_universes=n_universes,
            d=len(train_paths),
            f=ff.train_ensemble,
            f_args={
                "paths_to_encoded_datasets": train_paths,
                "train_index": indcs_train,
                "base_clf": MODEL[wildcards.model],
                "meta_clf": META_MODEL[wildcards.meta_model]
            },
            p=[p_0, 1 - p_0],
            funker_name=None,
            new_random_state_each_generation=False,
            n_jobs=n_universes,
            log_path=os.path.dirname(output[2]) + "/",
            log_file_name=os.path.basename(output[2])
        )

        best_solution, _ = mvo.run(0, max_iterations=max_generations, parallel=True)

        train_paths_best = np.array(train_paths)[np.nonzero(best_solution)[0]]

        # keep ensemble best encodings position for later usage
        indices = df_points.loc[
            df_points.encoding_1.isin(train_paths_best) &
            df_points.encoding_2.isin(train_paths_best)
        ].index
        df_points["ensemble_mvo"] = False
        df_points.iloc[indices, df_points.columns.get_loc("ensemble_mvo")] = True

        encoded_datasets = [pd.read_csv(p, index_col=0) for p in train_paths_best]

        X_train_list, X_test_list = \
            [df.loc[indcs_train, :].iloc[:, :-1].values
             for df in encoded_datasets], \
            [df.loc[indcs_test, :].iloc[:, :-1].values
             for df in encoded_datasets]

        y_train, y_test = \
            encoded_datasets[0].loc[indcs_train, "y"].values, \
            encoded_datasets[0].loc[indcs_test, "y"].values

        clf = MODEL[wildcards.model]
        eclf = META_MODEL[wildcards.meta_model]
        eclf.estimators = [(train_paths[i], clf) for i in range(len(train_paths_best))]

        try:
            eclf.fit(X_train_list, y_train)
            y_pred = eclf.predict(X_test_list)
            mcc = matthews_corrcoef(y_test,y_pred)
        except np.linalg.LinAlgError as e:
            print(e)
        except ValueError as e:
            print(e)

        pd.DataFrame({
            "mcc": [mcc],
            "fold": [wildcards.fold],
            "model": [wildcards.model],
            "meta_model": [wildcards.meta_model]
        }).to_csv(output[0])

        df_points.to_csv(output[1])


# def combine_point_data(lst_in, file_out):
#     df_res = pd.concat([pd.read_csv(p, index_col=0) for p in lst_in], axis=1, join="inner")
#     df_res = df_res.loc[:, ~df_res.columns.duplicated()].copy()
#     # close the path
#     df_tmp = df_res.loc[df_res.chull_complete == 0].copy()
#     df_tmp["chull_complete"] = df_res.chull_complete.sort_values(ascending=False).unique()[0] + 1
#     df_res = pd.concat([df_res, df_tmp])
#     df_res.to_csv(file_out)
#
# rule combine_point_data_0_4:
#     input:
#         "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_chull/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_pfront/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_mvo/{meta_model}/{model}/kappa_error_{fold}.csv"
#     output:
#         "data/temp/{dataset}/kappa_error_res/{meta_model}/{model}/{fold,[0-4]}.csv"
#     run:
#         combine_point_data(list(input), output[0])

# rule combine_point_data_5_99:
#     input:
#         "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_chull/{meta_model}/{model}/kappa_error_{fold}.csv",
#         "data/temp/{dataset}/ensemble_pfront/{meta_model}/{model}/kappa_error_{fold}.csv"
#     output:
#         "data/temp/{dataset}/kappa_error_res/{meta_model}/{model}/{fold,[5-9]|\d\d}.csv"
#     run:
#         combine_point_data(list(input), output[0])

# since encodings in best ensemble can vary across folds
# find the best common one across all folds
# rule get_single_best_encoding:
#     input:
#         "data/temp/{dataset}/single_encodings/{model}/res.csv",
#         lambda wildcards:
#             expand(f"data/temp/{wildcards.dataset}/ensemble_bst/"
#                    f"{wildcards.meta_model}/{wildcards.model}/kappa_error_{{fold}}.csv",
#                    fold=FOLDS),
#     output:
#         "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/single_best_enc.csv"
#     run:
#         df_encs = pd.read_csv(input[0], index_col=0)
#
#         df_res = pd.DataFrame()
#         for idx, p in enumerate(sorted(input[1:])):
#             df_tmp = pd.read_csv(p,index_col=0)
#             df_tmp["fold"] = idx
#             df_res = pd.concat([df_res, df_tmp])
#
#         all_encs = set(df_res[["encoding_1", "encoding_2"]].values.flatten())
#
#         df_bin = pd.DataFrame(
#             np.zeros((len(all_encs), len(FOLDS))),
#             index=all_encs,
#             columns=[f"fold_{i}" for i in FOLDS]
#         )
#
#         for i in df_bin.index:
#             fold_positions = df_res\
#                 .loc[((df_res.encoding_1 == i) | (df_res.encoding_2 == i)) & df_res.ensemble_best]["fold"]\
#                 .unique()
#             if len(fold_positions) > 0:
#                 df_bin.iloc[df_bin.index.get_loc(i), fold_positions] = 1
#
#         # common_encs = df_bin.iloc[np.nonzero((df_bin.sum(axis=1) == len(FOLDS)).values)].index
#         common_encs = df_bin.sum(axis=1).sort_values(ascending=False).index[:5]
#         common_encs = [e.split("/")[-1] for e in common_encs]
#
#         df_best = df_encs\
#             .loc[df_encs.encoding.isin(common_encs)]\
#             .groupby(["encoding"])["mcc"]\
#             .mean().sort_values(ascending=False).reset_index()
#
#         df_encs[df_encs.encoding == df_best.encoding[0]].to_csv(output[0])

# rule get_random_best_encoding:
#     input:
#         "data/temp/{dataset}/single_encodings/{model}/res.csv",
#         lambda wildcards:
#             expand(f"data/temp/{wildcards.dataset}/ensemble_rnd/"
#                    f"{wildcards.meta_model}/{wildcards.model}/kappa_error_{{fold}}.csv",
#                    fold=FOLDS)
#     output:
#         "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/single_rand_enc.csv"
#     run:
#         df_encs = pd.read_csv(input[0], index_col=0)
#
#         df_points = pd.concat([pd.read_csv(p, index_col=0) for p in input[1:]])
#
#         random_encodings = set(
#             df_points[df_points.ensemble_rand][["encoding_1", "encoding_2"]].values.flatten()
#         )
#
#         df_best = df_encs\
#             .loc[df_encs.encoding.isin([e.split("/")[-1] for e in random_encodings])]\
#             .groupby("encoding")["mcc"]\
#             .mean().sort_values(ascending=False)\
#             .reset_index()
#
#         df_encs[df_encs.encoding == df_best.encoding[0]].to_csv(output[0])

rule collect_ensembles:
    input:
        best=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensemble_bst/"
                   f"{wildcards.meta_model}/{wildcards.model}/{{fold}}.csv",
                   fold=FOLDS),
        rand=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensemble_rnd/"
                   f"{wildcards.meta_model}/{wildcards.model}/{{fold}}.csv",
                   fold=FOLDS),
        chull=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensemble_chull/"
               f"{wildcards.meta_model}/{wildcards.model}/{{fold}}.csv",
               fold=FOLDS),
        pfront=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensemble_pfront/"
                   f"{wildcards.meta_model}/{wildcards.model}/{{fold}}.csv",
                   fold=FOLDS),
        mvo=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensemble_mvo/"
                   f"{wildcards.meta_model}/{wildcards.model}/{{fold}}.csv",
                   fold=FOLDS[:5]),
        # single_best=
        #     "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/single_best_enc.csv",
        # rand_single_best=
        #     "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/single_rand_enc.csv"
    output:
        "data/temp/{dataset}/ensembles_res/{meta_model}/{model}/res.csv"
    run:
        df_res = pd.DataFrame()
        for k, path_obj in input.items():
            if type(path_obj) == snakemake.io.Namedlist:
                for p in path_obj:
                    df_tmp = pd.read_csv(p, index_col=0)
                    df_tmp["cat"] = k
                    df_res = pd.concat([df_res, df_tmp])
            else:
                df_tmp = pd.read_csv(path_obj, index_col=0)
                df_tmp["meta_model"] = wildcards.meta_model
                df_tmp["cat"] = k
                df_tmp.drop("encoding",axis=1,inplace=True)
                df_res = pd.concat([df_res, df_tmp])

        df_res.to_csv(output[0])

rule collect_all:
    input:
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensembles_res/{{meta_model}}/{{model}}/res.csv",
                   model=MODELS, meta_model=META_MODELS)
    output:
        "data/temp/{dataset}/ensembles_res/res.csv"
    run:
        df_res = pd.DataFrame()
        for  p in list(input):
            df_tmp = pd.read_csv(p, index_col=0)
            df_res = pd.concat([df_res, df_tmp])

        df_res.reset_index(drop=True).to_csv(output[0])

rule collect_all_remove_mvo:
    input:
        "data/temp/{dataset}/ensembles_res/res.csv"
    output:
        "data/temp/{dataset}/ensembles_res/res_wo_mvo.csv"
    run:
        df = pd.read_csv(input[0], index_col=0)
        df.drop(df.loc[df.cat == "mvo"].index, inplace=True)
        df.to_csv(output[0])

rule critical_difference:
    input:
        "data/temp/{dataset}/ensembles_res/res_wo_mvo.csv"
    output:
        "data/temp/{dataset}/ensembles_res/cd.yaml"
    script:
        "scripts/cd.R"

# rule plot_gens_vs_perf:
#     input:
#         lambda wildcards:
#                 expand(f"data/temp/{wildcards.dataset}/ensemble_mvo/{{meta_model}}/{{model}}/gens_vs_perf_{{fold}}.txt",
#                        meta_model=META_MODELS, model=MODELS, fold=FOLDS[:5])
#     output:
#         "data/temp/{dataset}/vis/gen_vs_perf.html"
#     run:
#         res2 = []
#         for p in list(input):
#             mmodel = p.split("/")[4]
#             model = p.split("/")[5]
#             fold = int(p[-5:-4])
#             with open(p) as f:
#                 res = list(chunked(f.readlines(),6))
#                 for idx, l in enumerate(res):
#                     fitness, mcc = l[2].rstrip().split(",")
#                     fitness = float(fitness.replace("Best Fitness: ",""))
#                     mcc = float(mcc.replace(" best metrics: {'mcc': ","").replace("}",""))
#                     res2.append([idx, fitness, mcc, fold, model, mmodel])
#
#         source = pd.DataFrame(res2,columns=["gen", "fitness", "mcc", "fold", "model", "mmodel"])
#
#         line = alt.Chart(source).mark_line(color="black").encode(
#             x="gen:O",
#             y="mean(fitness):Q"
#         )
#
#         band = alt.Chart(source).mark_errorband(extent="ci", color="black").encode(
#             x=alt.X("gen:O",title="Generation"),
#             y=alt.Y("fitness:Q",title="Fitness (1-MCC)")
#         )
#
#         (band + line).properties(
#             height=150,
#             width=150
#         ).facet(
#             column=alt.Column("model:N",title="Model"),
#             row=alt.Row("mmodel:N",title="Ensemble")
#         ).save(output[0])
#
# rule kappa_error_plot_data:
#     input:
#         lambda wildcards:
#             expand(f"data/temp/{wildcards.dataset}/kappa_error_res/stacking/{{model}}/{{fold}}.csv",
#                    fold=FOLDS, model=MODELS)
#     output:
#         "data/temp/{dataset}/kappa_error_res/plot_data.csv"
#     run:
#         # we use only one ensemble method here, because it does not influence the kappa-error values
#         df_res = pd.DataFrame()
#         for m in MODELS:
#             for f in FOLDS:
#                 path = [p for p in list(input) if f"/{m}/" in p and f"/{f}.csv" in p][0]
#                 df_tmp = pd.read_csv(path, index_col=0)
#                 filter_vals = [
#                     df_tmp.ensemble_best,
#                     df_tmp.ensemble_rand,
#                     df_tmp.ensemble_chull,
#                     df_tmp.ensemble_pfront
#                 ]
#                 if "ensemble_mvo" in df_tmp.columns:
#                     filter_vals.append(df_tmp.ensemble_mvo)
#                 filter_ = reduce(
#                     lambda v1, v2: v1 | v2,
#                     filter_vals[:-1],
#                     filter_vals[-1]
#                 )
#                 df_tmp1 = df_tmp \
#                     .loc[np.bitwise_not(filter_) & (df_tmp.chull_complete == -1)] \
#                     .sample(1000).copy()
#                 df_tmp2 = df_tmp \
#                     .loc[filter_ | (df_tmp.chull_complete != -1)].copy()
#                 df_tmp = pd.concat([df_tmp1, df_tmp2])
#                 df_tmp["model"], df_tmp["fold"] = m, f
#                 df_res = pd.concat([df_res, df_tmp])
#
#         df_res.loc[df_res.ensemble_mvo.isna(), "ensemble_mvo"] = False
#
#         df_res["cat"] = df_res.apply(
#             lambda row:
#                 "mvo" if row.ensemble_mvo else
#                 "chull" if row.ensemble_chull else
#                 "pfront" if row.ensemble_pfront else
#                 "best" if row.ensemble_best else
#                 "rand" if row.ensemble_rand else
#                 "all"
#             , axis=1)
#
#         df_res.to_csv(output[0])
#
# rule remove_mvo_kappa_error_plot_data:
#     input:
#         "data/temp/{dataset}/kappa_error_res/plot_data.csv"
#     output:
#         "data/temp/{dataset}/kappa_error_res/plot_data_wo_mvo.csv"
#     run:
#         df = pd.read_csv(input[0], index_col=0)
#         df.drop(df.loc[df.cat == "mvo"].index, inplace=True)
#         df.drop("ensemble_mvo", axis=1, inplace=True)
#         df.to_csv(output[0])
#
# rule kappa_error_plot:
#     input:
#         "data/temp/{dataset}/kappa_error_res/plot_data.csv"
#     output:
#         "data/temp/{dataset}/vis/kappa_error_plot.html"
#     run:
#         df_res = pd.read_csv(input[0], index_col=0)
#
#         scatter = alt.Chart().mark_point(filled=True).encode(
#             x=alt.X("x:Q", title="kappa", axis=None),
#             y=alt.Y("y:Q", title="average pair-wise error", axis=alt.Axis(grid=False)),
#             shape=alt.Shape(
#                 "cat:N", title="Pruning",
#                 legend=alt.Legend(offset=-45)
#             ),
#             color=alt.condition(
#                 alt.datum.cat == "all",
#                 alt.value("gray"),
#                 alt.value("black")
#             ),
#             size=alt.condition(
#                 alt.datum.cat == "all",
#                 alt.value(5),
#                 alt.value(65)
#             ),
#             opacity=alt.condition(
#                 alt.datum.cat == "all",
#                 alt.value(0.3),
#                 alt.value(0.7)
#             )
#         ).properties(
#             width=200,
#             height=200
#         )
#
#         convex_hull = alt.Chart().mark_line(
#             color="black",
#             size=1.1
#         ).encode(
#             x=alt.X("x:Q", title=None),
#             y=alt.Y("y:Q", title="average pair-wise error"),
#             order="chull:N",
#         ).transform_filter(
#             alt.datum.chull != -1
#         )
#
#         pareto_frontier = alt.Chart().mark_line(
#             strokeDash=[5, 1],
#             color="red",
#             size=1.1
#         ).encode(
#             x="x:Q",
#             y="y:Q",
#             order="pfront:N"
#         ).transform_filter(
#             alt.datum.pfront != -1
#         )
#
#         vals = np.array(range(51)) / 100
#         df = pd.DataFrame({"x": [1 - (1 / (1 - i)) for i in vals], "y": vals})
#         bound_line = alt.Chart(df).mark_line(color="lightgray").encode(
#             x="x:Q",
#             y="y:Q"
#         )
#
#         c1 = alt.layer(
#             scatter,
#             convex_hull,
#             pareto_frontier,
#             bound_line,
#             data=df_res.loc[df_res.fold == 0]
#         ).facet(
#             column=alt.Column("model", title=None),
#             spacing=0
#         )
#
#         heatmap = alt.Chart().mark_rect().encode(
#             x=alt.X(
#                 "x:Q",
#                 title="kappa",
#                 bin=alt.Bin(maxbins=40),
#                 axis=alt.Axis(values=[-1.0, -0.5, 0.0, 0.5, 1.0], format=".1f"),
#             ),
#             y=alt.Y(
#                 "y:Q",
#                 title="average pair-wise error",
#                 bin=alt.Bin(maxbins=40),
#                 axis=alt.Axis(values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], format=".1f"),
#             ),
#             color=alt.Color(
#                 "count(x):Q",
#                 title="Count",
#                 legend=alt.Legend(
#                     gradientLength=90,
#                     # values=[0, 1500, 3000, 4500]
#                     # values=[0, np.histogram2d(x=df_res.x, y=df_res.y, bins=45)[0].max()]
#                 ),
#                 scale=alt.Scale(scheme="greys")
#             ),
#             tooltip="count(x):Q"
#         ).properties(
#             height=200,
#             width=201
#         )
#
#         c2 = alt.layer(
#             heatmap,
#             bound_line,
#             data=df_res.reset_index()
#         ).facet(
#             column=alt.Column(
#                 "model:N",
#                 title=None,
#                 header=alt.Header(labels=False)
#             ),
#             spacing=1
#         )
#
#         alt.vconcat(c1, c2, spacing=1).resolve_scale(
#             color="independent"
#         ).save(output[0])
#
# rule box_plot:
#     input:
#         lambda wildcards:
#             expand(f"data/temp/{wildcards.dataset}/ensembles_res/{{meta_model}}/{{model}}/res.csv",
#                    meta_model=META_MODELS, model=MODELS)
#     output:
#         "data/temp/{dataset}/vis/box_plot.html"
#     run:
#         df_res = pd.DataFrame()
#         for p in list(input):
#             df_tmp = pd.read_csv(p, index_col=0)
#             df_res = pd.concat([df_res, df_tmp])
#
#         alt.Chart(df_res).mark_boxplot(
#             size=6, color="#6c6c6c", opacity=1.0
#         ).encode(
#             x=alt.X("cat:N", axis=None),
#             y=alt.Y(
#                 "mcc:Q",
#                 scale=alt.Scale(domain=[0.0, 1.0]),
#                 axis=alt.Axis(values=[0.1, 0.3, 0.5, 0.7, 0.9])
#             ),
#             color=alt.Color("cat:N", title="Pruning")
#         ).properties(
#             width=100,
#             height=100
#         ).facet(
#             row=alt.Row("meta_model:N", title="Ensemble"),
#             column=alt.Column("model:N", title="Model"),
#             spacing=1
#         ).save(
#             output[0],
#             vegalite_version="5.1.0"
#         )
#
# rule xcd_plot:
#     input:
#         "data/temp/{dataset}/ensembles_res/cd.yaml",
#         "data/temp/{dataset}/ensembles_res/res_wo_mvo.csv"
#     output:
#         "data/temp/{dataset}/vis/xcd_plot.html"
#     run:
#         from scripts.xcd_plot import XCDChart
#
#         with open(input[0]) as f:
#             cd_data = yaml.safe_load(f)
#
#         df_res = pd.read_csv(input[1], index_col=0)
#
#         xcd_chart = XCDChart(ensemble_data=df_res, cd_data=cd_data)
#         xcd_chart.save(output[0])
#
# rule box_plot_manova:
#     input:
#         "data/temp/{dataset}/kappa_error_res/plot_data.csv"
#     output:
#         "data/temp/{dataset}/vis/box_plot_manova.html"
#     run:
#         df_res = pd.read_csv(input[0], index_col=0)
#
#         df_out = pd.DataFrame()
#         for m in df_res.model.unique():
#             df_tmp = df_res.loc[df_res.model == m]
#             df_tmp = df_tmp.loc[np.bitwise_not(
#                 df_tmp.ensemble_mvo |
#                 df_tmp.ensemble_best |
#                 df_tmp.ensemble_rand |
#                 df_tmp.ensemble_chull |
#                 df_tmp.ensemble_pfront
#             ) & (df_tmp.chull_complete == -1)]
#             df_tmp = pd.concat([
#                 pd.DataFrame({"variable": df_tmp.x, "type": "kappa", "model": m}),
#                 pd.DataFrame({"variable": df_tmp.y, "type": "error", "model": m})
#             ])
#             df_out = pd.concat([df_out, df_tmp])
#
#         alt.Chart(df_out).mark_boxplot(
#             color="grey",
#             size=15
#         ).encode(
#             x=alt.X("type:N", title=None, axis=None),
#             y=alt.Y("variable:Q", title=None),
#             color=alt.Color(
#                 "type:N", title="Type",
#                 scale=alt.Scale(scheme="greys")
#             ),
#             column=alt.Column("model:N", title="Model", spacing=2)
#         ).properties(
#             width=50,
#             height=100
#         ).save(output[0])
#
# rule statistics:
#     input:
#         "data/temp/{dataset}/kappa_error_res/plot_data_wo_mvo.csv",
#         lambda wildcards:
#             expand(f"data/temp/{wildcards.dataset}/areas/{{model}}/res.csv", model=MODELS)
#     output:
#         "data/temp/{dataset}/stats/kappa_error/manova_summary.csv",
#         "data/temp/{dataset}/stats/kappa_error/manova_summary_aov.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_kappa_summary_aov.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_kappa_tukey_hsd.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_error_summary_aov.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_error_tukey_hsd.csv",
#         "data/temp/{dataset}/stats/areas/anova_summary_aov.csv",
#         "data/temp/{dataset}/stats/areas/anova_tukey_hsd.csv"
#     script:
#         "scripts/statistics.R"
#
# rule statistics_table:
#     input:
#         "data/temp/{dataset}/stats/kappa_error/manova_summary.csv",
#         "data/temp/{dataset}/stats/kappa_error/manova_summary_aov.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_kappa_summary_aov.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_kappa_tukey_hsd.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_error_summary_aov.csv",
#         "data/temp/{dataset}/stats/kappa_error/anova_error_tukey_hsd.csv",
#         "data/temp/{dataset}/stats/areas/anova_summary_aov.csv",
#         "data/temp/{dataset}/stats/areas/anova_tukey_hsd.csv"
#     output:
#         "data/temp/{dataset}/stats/table.html"
#     run:
#         with open(output[0], "w") as f:
#             for p in sorted(input):
#                 df_tmp = pd.read_csv(
#                     p,index_col=0,
#                     converters={"term": lambda v: v.replace("df_res$","")}
#                 )
#                 df_tmp.fillna("-",inplace=True)
#                 exp = p.split("/")[-1][:-4]
#                 df_tmp["experiment"] = exp
#                 f.write(f"<h4>{exp}</h4>\n{df_tmp.to_html(col_space='70px')}\n")
#
# rule zip_files:
#     input:
#         "data/temp/{dataset}/stats/table.html",
#         "data/temp/{dataset}/vis/box_plot.html",
#         "data/temp/{dataset}/vis/box_plot_manova.html",
#         "data/temp/{dataset}/vis/kappa_error_plot.html",
#         "data/temp/{dataset}/vis/xcd_plot.html"
#     output:
#         temp("data/temp/{dataset}/{dataset}.zip"),
#         "data/temp/{dataset}/{dataset}_zipped.zip"
#     shell:
#         """
#         zip -q -j {output[0]} {input};
#         zip -q -j {output[1]} {output[0]};
#         """
#
# rule dataset_tables:
#     input:
#         expand("data/temp/{dataset}/ensembles_res/{meta_model}/{model}/res.csv",
#                dataset=DATASETS, meta_model=META_MODELS, model=MODELS)
#     output:
#         "data/temp/all_datasets/tables/dataset_tables.html"
#     run:
#         from scipy.stats import ttest_rel
#
#         def get_table(df_res):
#
#             df_stats = df_res \
#                 .groupby(["dataset", "model", "cat", "meta_model"])["mcc"] \
#                 .describe().reset_index() \
#                 .loc[:, ['dataset', 'model', 'cat', 'meta_model', 'mean', 'std']]
#
#             df_final = df_stats \
#                 .groupby(["dataset", "cat"]) \
#                 .apply(lambda df: df.sort_values("mean", ascending=False).iloc[0, :]) \
#                 .reset_index(drop=True)
#
#             df_final["anno"] = df_final[["mean", "std"]].apply(
#                 lambda row: f"{np.round(row[0], 2)} (±{np.round(row[1], 2)})",
#                 axis=1
#             )
#
#             df_out = df_final.pivot(index="dataset", columns="cat", values="anno")
#
#             for ds in df_final.dataset.unique():
#
#                 best_ens_cat, best_ens_mm, best_ens_m = df_final\
#                     .loc[df_final.dataset == ds]\
#                     .sort_values("mean", ascending=False)[["cat", "meta_model", "model"]]\
#                     .iloc[0]
#
#                 single_best_mm, single_best_m = df_final\
#                     .loc[(df_final.dataset == ds) & (df_final.cat == "single_best")]\
#                     .sort_values("mean",ascending=False)[["meta_model", "model"]] \
#                     .iloc[0]
#
#                 a1 = df_res.loc[
#                     (df_res.dataset == ds) &
#                     (df_res.cat == best_ens_cat) &
#                     (df_res.meta_model == best_ens_mm) &
#                     (df_res.model == best_ens_m)
#                 , "mcc"].values
#
#                 a2 = df_res.loc[
#                     (df_res.dataset == ds) &
#                     (df_res.cat == "single_best") &
#                     (df_res.meta_model == single_best_mm) &
#                     (df_res.model == single_best_m)
#                 , "mcc"
#                 ].values
#
#                 _, pval = ttest_rel(a1, a2, alternative="greater")
#
#                 # 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#                 if pval == 0:
#                     sig = "***"
#                 elif pval < 0.001:
#                     sig = "**"
#                 elif pval < 0.01:
#                     sig = "*"
#                 elif pval < 0.05:
#                     sig = "."
#                 else:
#                     sig = "ns"
#
#                 df_out.loc[ds, best_ens_cat] = \
#                     df_out.loc[ds, best_ens_cat].replace(" (",f"<sup>{sig}</sup> (")
#
#                 df_out.columns.name = None
#                 df_out.index.name = None
#
#             return df_out.to_html(escape=False)
#
#         df_res = pd.DataFrame()
#         for p in list(input):
#             df_tmp = pd.read_csv(p,index_col=0)
#             dataset = p.split("/")[2]
#             df_tmp["dataset"] = dataset
#             df_res = pd.concat([df_res, df_tmp])
#
#         t1 = get_table(df_res)
#
#         df_res = df_res.loc[df_res.model != "rf"]
#         t2 = get_table(df_res)
#
#         with open(output[0], "w") as f:
#             h1 = "<h3>With RF</h3>"
#             h2 = "<h3>Without RF</h3>"
#             f.write(f"{h1}\n{t1}\n{h2}\n{t2}\n")
#             f.flush()
#
# rule areas_table:
#     input:
#         expand("data/temp/{dataset}/areas/{model}/res.csv",
#                dataset=DATASETS, model=MODELS)
#     output:
#         "data/temp/all_datasets/tables/areas_table.html"
#     run:
#         df_stats = pd.DataFrame()
#         for p in list(input):
#             df_tmp = pd.read_csv(p, index_col=0)
#             df_res = df_tmp.area.describe()[["mean", "std"]]
#             df_res["dataset"] = p.split("/")[2]
#             df_res["model"] = p.split("/")[-2]
#             df_stats = pd.concat([
#                 df_stats,
#                 df_res.to_frame().transpose()
#             ])
#
#         df_stats.reset_index(drop=True, inplace=True)
#
#         df_stats["anno"] = df_stats[["mean", "std"]].apply(
#             lambda row: f"{np.round(row[0], 2)} (±{np.round(row[1], 3)})",
#             axis=1
#         )
#
#         df_out = df_stats.pivot(index="dataset", columns="model", values="anno")
#         df_out.columns.name = None
#         df_out.index.name = None
#
#         with open(output[0], "w") as f:
#             f.write(f"{df_out.to_html()}\n")
#             f.flush()
#
