import os
from typing import List, Tuple

import optuna
from numpy.lib.stride_tricks import sliding_window_view
from optuna import create_study
from scipy.signal import savgol_filter, find_peaks

from sklearn.model_selection import cross_val_score, KFold, BaseCrossValidator, GroupKFold
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from own_model.src.dataset_manip import load_data, state_change_eval, MyDataset


def get_peaks(x: pd.DataFrame):
    objectID = x.index.get_level_values(0).unique()
    x["smoothed"] = savgol_filter(x["Inclination (deg)_std"], 13, 3)

    inc_std_average = x["smoothed"].median()
    max_std = x["smoothed"].max()
    # print(f"ObjectID: {objectID}, Median: {inc_std_average} and max: {max_std}")

    peaks, _ = find_peaks(x["smoothed"], height=inc_std_average * 4)
    # print(peaks)

    x.loc[(objectID, peaks), "peak"] = 1
    return x


def create_triangle_kernel(period: int) -> np.array:
    length = 2 * period
    kernel = np.zeros(length)

    # The midpoint of the kernel, adding 1 to handle zero-based indexing
    midpoint = period

    # Increasing linear sequence up to the midpoint
    for i in range(midpoint):
        kernel[i] = (i + 1) / midpoint

    # Decreasing linear sequence after the midpoint
    for i in range(midpoint, length):
        kernel[i] = (length - i) / midpoint

    return kernel


def dynamic_weighting(x: pd.DataFrame):
    index = np.arange(x.shape[0])
    peak_index: np.array = index[x["peak"] == 1]

    # if we have 0 or just one peak we cannot calculate a period
    if peak_index.size <= 1:
        return x

    # Calculate distance between consecutive peaks
    distance = np.diff(peak_index)
    # Determine expected period, may be a bit broke if we only have a few distances
    period = int(np.median(distance))

    # if period is to large it would break our conv, and it's not really a period anyways
    if period * 2 >= x.shape[0]:
        return x

    triangle_kernel = create_triangle_kernel(period)
    peak_exp = np.convolve(x["peak"].to_numpy(), triangle_kernel, mode="same")
    x["peak_exp"] = peak_exp

    # x["peak_exp"] = x["peak"][::-1].rolling(window=period).max().bfill()[::-1]
    # x["peak_exp"] = x["peak_exp"].ewm(span=period).sum()

    return x


def add_engineered_peaks(df: pd.DataFrame, features_in: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    # already allocating features space
    features = features_in.copy()
    df["peak"] = 0.0
    df["peak_exp"] = 0.0
    df["smoothed"] = 0.0

    # get peaks
    df = df.groupby(level=0, group_keys=False).apply(get_peaks)
    # weight peaks
    df = df.groupby(level=0, group_keys=False).apply(dynamic_weighting)

    features.append("peak_exp")
    return df, features


def custom_loss_func(ground_truth: pd.Series, predictions: np.array, **kwargs):
    predicted_sum = np.convolve(predictions, np.ones(5), mode="same")
    predicted = np.zeros(predicted_sum.shape[0])
    predicted[predicted_sum >= 5] = 1

    ground_truth_sum = np.convolve(ground_truth, np.ones(5), mode="same")
    truth = np.zeros(ground_truth_sum.shape[0])
    truth[ground_truth_sum >= 5] = 1

    # f2 score
    f2_score = fbeta_score(truth, predicted, beta=2, zero_division=0.0)
    return f2_score


def add_lag_features(df: pd.DataFrame, feature_cols: List[str], lag_steps: int):
    new_columns = pd.DataFrame({f"{col}_lag{i}": df.groupby(level=0, group_keys=False)[col].shift(i * 3)
                                for i in range(1, lag_steps + 1)
                                for col in feature_cols}, index=df.index)
    new_columns_neg = pd.DataFrame({f"{col}_lag-{i}": df.groupby(level=0, group_keys=False)[col].shift(i * -3)
                                    for i in range(1, lag_steps + 1)
                                    for col in feature_cols}, index=df.index)
    df_out = pd.concat([df, new_columns, new_columns_neg], axis=1)
    features_out = feature_cols + new_columns.columns.tolist() + new_columns_neg.columns.to_list()
    # fill nans
    df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.bfill())
    df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.ffill())
    return df_out, features_out


def load_stuff():
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0
    # RF approach
    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW.copy() if DIRECTION == "EW" else BASE_FEATURES_NS.copy()

    # unwrap
    df[features] = np.unwrap(np.deg2rad(df[features]))

    # FEATURE ENGINEERING
    window_size = 6
    feature_dict = ENGINEERED_FEATURES_EW if DIRECTION == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()
            features.append(new_feature_name)

    # add engineered inclination feature
    df, features = add_engineered_peaks(df, features)

    df, features = add_lag_features(df, features, lag_steps=8)

    # # adding smoothing because of some FPs
    # new_feature_name = "Inclination (deg)" + "_" + "std"
    # df[new_feature_name] = df.groupby(level=0, group_keys=False)[["Inclination (deg)"]].apply(
    #     lambda x: x.rolling(window=6).std())
    # df[new_feature_name + "smoothed_1"] = df.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x[::-1].ewm(span=100, adjust=True).sum()[::-1])
    # df[new_feature_name + "smoothed_2"] = df.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x.ewm(span=100, adjust=True).sum())
    # df.bfill(inplace=True)
    # df.ffill(inplace=True)
    # df[new_feature_name + "_smoothed"] = (df[new_feature_name + "smoothed_1"] +
    #                                       df[new_feature_name + "smoothed_2"]) / 2
    # # df[new_feature_name + "smoothed"] = df.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    # # lambda x: x[::-1].rolling(window=170).mean()[::-1])
    # features.append(new_feature_name)
    # features.append(new_feature_name + "_smoothed")

    return df, features


def objective(trial):
    # learning_rate = trial.suggest_float("learning_rate", 0.1, 0.9, log=True)
    # n_estimators = trial.suggest_int("n_estimators", 50, 400)
    # reg_lambda = trial.suggest_float("reg_lambda", 0.01, 2.0, log=True)
    # max_depth = trial.suggest_int("max_depth", 2, 12)
    # gamma = trial.suggest_float("gamma", 0, 5)
    # subsample = trial.suggest_float("subsample", 0.5, 1.0)
    learning_rate = trial.suggest_categorical("learning_rate", [0.1, 0.3, 0.5, 0.7, 0.9])
    n_estimators = trial.suggest_categorical("max_iter", [50, 100, 150, 200, 300, 400])
    reg_lambda = trial.suggest_categorical("reg_lambda", [0.01, 0.1, 0.5, 1.0, 1.5, 2.0])
    max_depth = trial.suggest_categorical("max_depth", [10, 12, 14, 16, 18, 20, 24, 28, 32, 36])

    # rf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced", learning_rate=learning_rate,
    #                                     max_iter=max_iter, early_stopping=False, max_leaf_nodes=None,
    #                                     min_samples_leaf=25, l2_regularization=0.1)

    rf = XGBClassifier(random_state=RANDOM_STATE, n_estimators=n_estimators, max_leaves=0, learning_rate=learning_rate,
                       verbosity=2, tree_method="hist", scale_pos_weight=scale_pos, reg_lambda=reg_lambda,
                       max_depth=max_depth, n_jobs=15)

    f2_scorer = make_scorer(custom_loss_func, beta=2)
    # f2_scorer = make_scorer(fbeta_score, beta=2)

    groups = df.index.get_level_values(0).tolist()
    gkf = GroupKFold(n_splits=5)
    # this is the fastest and easiest way to get a continous index without destroying the pd.MultiIndex
    score = cross_val_score(rf, df[features], df[DIRECTION], groups=groups, n_jobs=1, scoring=f2_scorer, cv=gkf)  #
    f2_score = score.mean()
    return f2_score


if __name__ == "__main__":
    TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train")
    TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train_labels.csv")

    BASE_FEATURES_EW = [
        # "Eccentricity",
        # "Semimajor Axis (m)",
        # "Inclination (deg)",
        # "RAAN (deg)",
        # '"Argument of Periapsis (deg)",
        # "True Anomaly (deg)",
        # "Latitude (deg)",
        "Longitude (deg)",
        # "Altitude (m)",  # This is just first div of longitude?
    ]
    BASE_FEATURES_NS = [
        "Eccentricity",
        # "Semimajor Axis (m)",
        "Inclination (deg)",
        "RAAN (deg)",
        # "Argument of Periapsis (deg)",
        # "True Anomaly (deg)",
        # "Latitude (deg)",
        "Longitude (deg)",
        # "Altitude (m)",
        # "X (m)",
        # "Y (m)",
        # "Z (m)",
        # "Vx (m/s)",
        # "Vy (m/s)",
        # "Vz (m/s)"
    ]

    ENGINEERED_FEATURES_EW = {
        ("std", lambda x: x.rolling(window=6).std()):
            ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "Inclination (deg)"],  # , "RAAN (deg)"
    }
    ENGINEERED_FEATURES_NS = {
        ("std", lambda x: x.rolling(window=6).std()):
            ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "Inclination (deg)"],
        # "Semimajor Axis (m)", "Latitude (deg)", "RAAN (deg)", "Inclination (deg)"
    }

    DEG_FEATURES = [
        "Inclination (deg)",
        "RAAN (deg)",
        "Argument of Periapsis (deg)",
        "True Anomaly (deg)",
        "Latitude (deg)",
        "Longitude (deg)"
    ]

    RANDOM_STATE = 42
    DIRECTION = "NS"
    NUM_CSV_SETS = -1
    from pathlib import Path

    # fixing joblib windows bug
    base_dir_joblib_temp_folder = Path("C:/Users/nikla/Desktop/test_db/~joblib")
    base_dir_joblib_temp_folder.mkdir(exist_ok=True, parents=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(base_dir_joblib_temp_folder)

    # Beginning
    df, features = load_stuff()

    num_pos = df[DIRECTION].sum()
    all_samples = df[DIRECTION].shape[0]
    num_neg = all_samples - num_pos
    scale_pos = num_neg / num_pos

    study_name = "hSearch_XGBoost_NS_peaks"
    db_file_path = "C:/Users/nikla/Desktop/test_db/mystorage_GDB.db"
    # sampler = optuna.samplers.CmaEsSampler(seed=RANDOM_STATE)
    study = create_study(load_if_exists=True, study_name=study_name, direction="maximize",
                         storage=f'sqlite:///{db_file_path}')  # , sampler=sampler
    study.optimize(objective, n_trials=75)
    # dump(study, "study.pkl")
    # Print the optimal hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    print('Best params:', study.best_params)
    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial)
