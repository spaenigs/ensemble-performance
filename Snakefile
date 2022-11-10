import itertools
import os.path

import pandas as pd
import numpy as np

import functools
import math

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

from utils import get_csv_names
from optimizer.ensemble import StackingClassifier, VotingClassifier

from optimizer.optimization import BinaryMVO
import optimizer.optimization.fitness_function as ff

FOLDS = range(100)

MODEL = {
    "lr": LogisticRegression(),
    "dt": DecisionTreeClassifier(),
    "bayes": GaussianNB(),
    "rf":  RandomForestClassifier(),
    "mlp": MLPClassifier()
}
MODELS = list(MODEL.keys())

META_MODEL = {
    "stacking": StackingClassifier(estimators=None),
    "voting_soft": VotingClassifier(estimators=None, voting="soft"),
    "voting_hard": VotingClassifier(estimators=None, voting="hard")
}
META_MODELS = list(META_MODEL.keys())

DATASETS = [
    "avp_amppred",
    "amp_antibp2",
    "isp_il10pred",
    "cpp_mlcpp-complete",
    "nep_neuropipred-complete",
    "pip_pipel",
    "aip_antiinflam-complete",
    "acp_mlacp",
    "atb_antitbp",
    "hem_hemopi"
]

N_ENCODINGS = None

wildcard_constraints:
    fold="\d+"

rule all:
    input:
        # 1) prepare data and compute results
        expand("data/temp/{dataset}/single_encodings/{model}/res.csv",
            model=MODELS,dataset=DATASETS),
        expand("data/temp/{dataset}/kappa_error_all/{model}/{fold}.csv",
            dataset=DATASETS, model=MODELS,
               fold=FOLDS),
        expand("data/temp/{dataset}/ensembles_res/{meta_model}/{model}/res.csv",
               dataset=DATASETS, meta_model=META_MODELS, model=MODELS),
        expand("data/temp/{dataset}/areas/{model}/res.csv",
               dataset=DATASETS, model=MODELS),
        expand("data/temp/{dataset}/ensembles_res/res.csv",
            dataset=DATASETS),
        expand("data/temp/{dataset}/kappa_error_res/plot_data.csv", dataset=DATASETS),

        # 2) create plots
        expand("data/temp/{dataset}/vis/kappa_error_plot.html", dataset=DATASETS),
        expand("data/temp/{dataset}/vis/gen_vs_perf.{ftype}", dataset=DATASETS, ftype=["html", "png"]),
        expand("data/temp/{dataset}/vis/box_plot.{ftype}", dataset=DATASETS, ftype=["html", "png"]),
        expand("data/temp/{dataset}/vis/xcd_plot.{ftype}", dataset=DATASETS, ftype=["html", "png"]),
        expand("data/temp/{dataset}/vis/box_plot_manova.html", dataset=DATASETS),

        # 3) run statistics and create tables
        expand("data/temp/{dataset}/stats/table.html", dataset=DATASETS),

        # 4) misc
        expand("data/temp/{dataset}/encodings.csv", dataset=DATASETS),
        # expand("data/temp/{dataset}/data_temp_{dataset}.tar.gz", dataset=DATASETS),

        # 5) Combine results
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

        res = df_res.groupby("encoding")\
            .apply(lambda df: df.mcc.mean())\
            .sort_values(ascending=False)

        top3 = res.index[:3].to_list()

        df_res = df_res.loc[df_res.encoding.isin(top3)]
        df_res["rank"] = -1
        df_res["cat"] = "single"

        for i, enc in enumerate(top3, start=1):
            df_res.loc[df_res.encoding == enc, "rank"] = f"Top_{i}"

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

        df_points = pd.read_csv(input[3], index_col=0)

        # y is average pairwise error
        train_paths = list(set(
            df_points[["encoding_1", "encoding_2"]] \
                .values.flatten()
        ))

        n_universes = 32
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


def combine_point_data(lst_in, file_out):
    df_res = pd.concat([pd.read_csv(p, index_col=0) for p in lst_in], axis=1, join="inner")
    df_res = df_res.loc[:, ~df_res.columns.duplicated()].copy()
    # close the path
    df_tmp = df_res.loc[df_res.chull_complete == 0].copy()
    df_tmp["chull_complete"] = df_res.chull_complete.sort_values(ascending=False).unique()[0] + 1
    df_res = pd.concat([df_res, df_tmp])
    df_res.to_csv(file_out)


rule combine_point_data_0_4:
    input:
        "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_chull/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_pfront/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_mvo/{meta_model}/{model}/kappa_error_{fold}.csv"
    output:
        "data/temp/{dataset}/kappa_error_res/{meta_model}/{model}/{fold,[0-4]}.csv"
    run:
        combine_point_data(list(input), output[0])

rule combine_point_data_5_99:
    input:
        "data/temp/{dataset}/ensemble_bst/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_rnd/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_chull/{meta_model}/{model}/kappa_error_{fold}.csv",
        "data/temp/{dataset}/ensemble_pfront/{meta_model}/{model}/kappa_error_{fold}.csv"
    output:
        "data/temp/{dataset}/kappa_error_res/{meta_model}/{model}/{fold,[5-9]|\d\d}.csv"
    run:
        combine_point_data(list(input), output[0])

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

rule critical_difference:
    input:
        lambda wildcards:
                expand(f"data/temp/{wildcards.dataset}/ensembles_res/{{meta_model}}/{{model}}/res.csv",
                    model=MODELS, meta_model=META_MODELS)
        # "data/temp/{dataset}/ensembles_res/res_wo_mvo.csv"
    output:
        "data/temp/{dataset}/ensembles_res/cd.yaml"
    script:
        "scripts/cd.R"

rule kappa_error_plot_data:
    input:
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/kappa_error_res/stacking/{{model}}/{{fold}}.csv",
                   fold=FOLDS, model=MODELS)
    output:
        "data/temp/{dataset}/kappa_error_res/plot_data.csv"
    run:
        # we use only one ensemble method here, because it does not influence the kappa-error values
        df_res = pd.DataFrame()
        for m in MODELS:
            for f in FOLDS:
                path = [p for p in list(input) if f"/{m}/" in p and f"/{f}.csv" in p][0]
                df_tmp = pd.read_csv(path, index_col=0)
                filter_vals = [
                    df_tmp.ensemble_best,
                    df_tmp.ensemble_rand,
                    df_tmp.ensemble_chull,
                    df_tmp.ensemble_pfront
                ]
                if "ensemble_mvo" in df_tmp.columns:
                    filter_vals.append(df_tmp.ensemble_mvo)
                filter_ = reduce(
                    lambda v1, v2: v1 | v2,
                    filter_vals[:-1],
                    filter_vals[-1]
                )
                df_tmp1 = df_tmp \
                    .loc[np.bitwise_not(filter_) & (df_tmp.chull_complete == -1)] \
                    .sample(1000).copy()
                df_tmp2 = df_tmp \
                    .loc[filter_ | (df_tmp.chull_complete != -1)].copy()
                df_tmp = pd.concat([df_tmp1, df_tmp2])
                df_tmp["model"], df_tmp["fold"] = m, f
                df_res = pd.concat([df_res, df_tmp])

        df_res.loc[df_res.ensemble_mvo.isna(), "ensemble_mvo"] = False

        df_res["cat"] = df_res.apply(
            lambda row:
                "mvo" if row.ensemble_mvo else
                "chull" if row.ensemble_chull else
                "pfront" if row.ensemble_pfront else
                "best" if row.ensemble_best else
                "rand" if row.ensemble_rand else
                "all"
            , axis=1)

        df_res.to_csv(output[0])

rule kappa_error_plot:
    input:
        "data/temp/{dataset}/kappa_error_res/plot_data.csv"
    output:
        "data/temp/{dataset}/vis/kappa_error_plot.html"
    script:
        "scripts/plots/kappa_error.py"

rule plot_gens_vs_perf:
    input:
        lambda wildcards:
                expand(f"data/temp/{wildcards.dataset}/ensemble_mvo/{{meta_model}}/{{model}}/gens_vs_perf_{{fold}}.txt",
                       meta_model=META_MODELS, model=MODELS, fold=FOLDS[:5])
    output:
        "data/temp/{dataset}/vis/gen_vs_perf.html",
        "data/temp/{dataset}/vis/gen_vs_perf.png"
    script:
        "scripts/plots/gens_vs_perf.py"

rule box_plot:
    input:
        ensemble_res=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensembles_res/{{meta_model}}/{{model}}/res.csv",
                   meta_model=META_MODELS, model=MODELS),
        single_res=lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/single_encodings/{{model}}/res.csv",
                   model=MODELS)
    output:
        "data/temp/{dataset}/vis/box_plot.html",
        "data/temp/{dataset}/vis/box_plot.png"
    script:
        "scripts/plots/box_plot.py"

rule xcd_plot:
    input:
        "data/temp/{dataset}/ensembles_res/cd.yaml",
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/ensembles_res/{{meta_model}}/{{model}}/res.csv",
                   meta_model=META_MODELS, model=MODELS)
    output:
        "data/temp/{dataset}/vis/xcd_plot.html",
        "data/temp/{dataset}/vis/xcd_plot.png"
    script:
        "scripts/plots/xcd.py"

rule box_plot_manova:
    input:
        "data/temp/{dataset}/kappa_error_res/plot_data.csv"
    output:
        "data/temp/{dataset}/vis/box_plot_manova.html"
    script:
        "scripts/plots/box_plot_manova.py"

rule statistics:
    input:
        "data/temp/{dataset}/kappa_error_res/plot_data.csv",
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/areas/{{model}}/res.csv", model=MODELS)
    output:
        "data/temp/{dataset}/stats/kappa_error/manova_summary.csv",
        "data/temp/{dataset}/stats/kappa_error/manova_summary_aov.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_kappa_summary_aov.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_kappa_tukey_hsd.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_error_summary_aov.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_error_tukey_hsd.csv",
        "data/temp/{dataset}/stats/areas/anova_summary_aov.csv",
        "data/temp/{dataset}/stats/areas/anova_tukey_hsd.csv"
    script:
        "scripts/statistics.R"

rule statistics_table:
    input:
        "data/temp/{dataset}/stats/kappa_error/manova_summary.csv",
        "data/temp/{dataset}/stats/kappa_error/manova_summary_aov.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_kappa_summary_aov.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_kappa_tukey_hsd.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_error_summary_aov.csv",
        "data/temp/{dataset}/stats/kappa_error/anova_error_tukey_hsd.csv",
        "data/temp/{dataset}/stats/areas/anova_summary_aov.csv",
        "data/temp/{dataset}/stats/areas/anova_tukey_hsd.csv"
    output:
        "data/temp/{dataset}/stats/table.html"
    run:
        with open(output[0], "w") as f:
            for p in sorted(input):
                df_tmp = pd.read_csv(
                    p,index_col=0,
                    converters={"term": lambda v: v.replace("df_res$","")}
                )
                df_tmp.fillna("-",inplace=True)
                exp = p.split("/")[-1][:-4]
                df_tmp["experiment"] = exp
                f.write(f"<h4>{exp}</h4>\n{df_tmp.to_html(col_space='70px')}\n")

rule encodings_table:
    input:
        lambda wildcards:
            expand(f"data/{wildcards.dataset}/csv/all/{{csv_name}}",
                   csv_name=get_csv_names(wildcards.dataset)[:N_ENCODINGS])
    output:
        "data/temp/{dataset}/encodings.csv"
    run:
        paths = [p.split("/")[-1].replace(".csv", "") for p in list(input)]
        paths = sorted(set(paths))
        arr = ["electrostatic_hull", "dist_freq"]
        paths = [[p[:18]] + p[18:].split("_")
                 if "hull" in p else [p[:9]] + p[10:].split("_")
                 if "dist_freq" in p else p.split("_") for p in paths]
        paths = [{k: v for k,v in zip(range(len(p)), p)} for p in paths]

        df = pd.DataFrame(paths)
        df.columns = ["param_" + str(i) for i in range(5)]


        def calc(vals):
            return "; ".join([str(v) for v in set(vals)])


        df = df.groupby("param_0").apply(lambda df: pd.DataFrame({
            "params_1": [calc(df["param_1"].values)],
            "params_2": [calc(df["param_2"].values)],
            "params_3": [calc(df["param_3"].values)],
            "params_4": [calc(df["param_4"].values)],
            #"params_5": [calc(df["param_5"].values)],
        })).reset_index(drop=False)

        df = df.drop(["level_1"], axis=1)

        cols = list(df.columns)
        cols[0] = "encoding"
        df.columns = cols

        df = df.replace("nan", "")

        df.to_csv(output[0], sep=",", index_label=False)

rule zip_files:
    input:
        lambda wildcards:
            expand(f"data/temp/{wildcards.dataset}/single_encodings/{{model}}/res.csv", model=MODELS),
        "data/temp/{dataset}/stats/",
        "data/temp/{dataset}/vis/",
        "data/temp/{dataset}/ensembles_res/",
        "data/temp/{dataset}/areas/",
        "data/temp/{dataset}/encodings.csv"
    output:
        "data/temp/{dataset}/data_temp_{dataset}.tar.gz"
    shell:
        "tar czf {output[0]} {input}"

rule dataset_tables:
    input:
        ensembles_res=expand(
            "data/temp/{dataset}/ensembles_res/{meta_model}/{model}/res.csv",
            dataset=DATASETS, meta_model=META_MODELS, model=MODELS),
        single_encodings_res=expand(
            "data/temp/{dataset}/single_encodings/{model}/res.csv",
            dataset=DATASETS, model=MODELS)
    output:
        "data/temp/all_datasets/tables/dataset_tables.html"
    run:
        from scipy.stats import ttest_rel

        def get_table(df_res):

            df_stats = df_res \
                .groupby(["dataset", "model", "cat", "meta_model"])["mcc"] \
                .describe().reset_index() \
                .loc[:, ['dataset', 'model', 'cat', 'meta_model', 'mean', 'std']]

            df_final = df_stats \
                .groupby(["dataset", "cat"]) \
                .apply(lambda df: df.sort_values("mean", ascending=False).iloc[0, :]) \
                .reset_index(drop=True)

            df_final["anno"] = df_final[["mean", "std"]].apply(
                lambda row: f"{np.round(row[0], 2)} (±{np.round(row[1], 2)})",
                axis=1
            )

            df_out = df_final.pivot(index="dataset", columns="cat", values="anno")

            for ds in df_final.dataset.unique():

                best_ens_cat, best_ens_mm, best_ens_m = df_final\
                    .loc[df_final.dataset == ds]\
                    .sort_values("mean", ascending=False)[["cat", "meta_model", "model"]]\
                    .iloc[0]

                single_best_mm, single_best_m = df_final\
                    .loc[(df_final.dataset == ds) & (df_final.cat == "single")]\
                    .sort_values("mean",ascending=False)[["meta_model", "model"]] \
                    .iloc[0]

                a1 = df_res.loc[
                    (df_res.dataset == ds) &
                    (df_res.cat == best_ens_cat) &
                    (df_res.meta_model == best_ens_mm) &
                    (df_res.model == best_ens_m)
                , "mcc"].values

                a2 = df_res.loc[
                    (df_res.dataset == ds) &
                    (df_res.cat == "single") &
                    (df_res.meta_model == single_best_mm) &
                    (df_res.model == single_best_m)
                , "mcc"
                ].values[:len(a1)]  # in case MVO is best method

                _, pval = ttest_rel(a1, a2, alternative="greater")

                # 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
                if pval == 0:
                    sig = "***"
                elif pval < 0.001:
                    sig = "**"
                elif pval < 0.01:
                    sig = "*"
                elif pval < 0.05:
                    sig = "."
                else:
                    sig = "ns"

                df_out.loc[ds, best_ens_cat] = \
                    df_out.loc[ds, best_ens_cat].replace(" (",f"<sup>{sig}</sup> (")

                df_out.columns.name = None
                df_out.index.name = None

            return df_out.to_html(escape=False)

        df_res = pd.DataFrame()

        for p in list(input.ensembles_res):
            df_tmp = pd.read_csv(p,index_col=0)
            dataset = p.split("/")[2]
            df_tmp["dataset"] = dataset
            df_res = pd.concat([df_res, df_tmp])

        for p in list(input.single_encodings_res):
            df_tmp = pd.read_csv(p,index_col=0)
            dataset = p.split("/")[2]
            df_tmp["dataset"] = dataset
            df_tmp = df_tmp.loc[df_tmp["rank"] == "Top_1", :]
            df_tmp = df_tmp.drop(["encoding"],axis=1)
            df_tmp = df_tmp.rename(columns={"rank": "meta_model"})
            df_res = pd.concat([df_res, df_tmp])

        t1 = get_table(df_res)

        df_res = df_res.loc[df_res.model != "rf"]
        t2 = get_table(df_res)

        with open(output[0], "w") as f:
            h1 = "<h3>With RF</h3>"
            h2 = "<h3>Without RF</h3>"
            f.write(f"{h1}\n{t1}\n{h2}\n{t2}\n")
            f.flush()

rule areas_table:
    input:
        expand("data/temp/{dataset}/areas/{model}/res.csv",
               dataset=DATASETS, model=MODELS)
    output:
        "data/temp/all_datasets/tables/areas_table.html"
    run:
        df_stats = pd.DataFrame()
        for p in list(input):
            df_tmp = pd.read_csv(p, index_col=0)
            df_res = df_tmp.area.describe()[["mean", "std"]]
            df_res["dataset"] = p.split("/")[2]
            df_res["model"] = p.split("/")[-2]
            df_stats = pd.concat([
                df_stats,
                df_res.to_frame().transpose()
            ])

        df_stats.reset_index(drop=True, inplace=True)

        df_stats["anno"] = df_stats[["mean", "std"]].apply(
            lambda row: f"{np.round(row[0], 2)} (±{np.round(row[1], 3)})",
            axis=1
        )

        df_out = df_stats.pivot(index="dataset", columns="model", values="anno")
        df_out.columns.name = None
        df_out.index.name = None

        with open(output[0], "w") as f:
            f.write(f"{df_out.to_html()}\n")
            f.flush()
