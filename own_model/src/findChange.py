import math
import os
import warnings
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from scipy.signal import find_peaks_cwt, find_peaks, peak_prominences, savgol_filter

from own_model.src.dataset_manip import load_data, state_change_eval, MyDataset
from own_model.src.myModel import LitChangePointClassifier
from baseline_submissions.evaluation import NodeDetectionEvaluator


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


def add_lag_features(df: pd.DataFrame, feature_cols: List[str], lag_steps: int):
    new_columns = pd.DataFrame({f"{col}_lag{i}": df.groupby(level=0, group_keys=False)[col].shift(i * 3)  #
                                for i in range(1, lag_steps + 1)
                                for col in feature_cols}, index=df.index)
    new_columns_neg = pd.DataFrame({f"{col}_lag-{i}": df.groupby(level=0, group_keys=False)[col].shift(i * -3)  #
                                    for i in range(1, lag_steps + 1)
                                    for col in feature_cols}, index=df.index)
    df_out = pd.concat([df, new_columns, new_columns_neg], axis=1)
    features_out = feature_cols + new_columns.columns.tolist() + new_columns_neg.columns.to_list()
    # fill nans
    # df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.bfill())
    # df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.ffill())
    df_out.fillna(0, inplace=True)
    return df_out, features_out


def print_params(features, direction):
    print("PARAMS:")
    print(f"NUM CSVs: {NUM_CSV_SETS}")
    print(f"DIRECTION: {direction}")
    if direction == "EW":
        print(f"BASE_FEATURES: {BASE_FEATURES_EW}")
        print(f"ENGINEERED FEATURES: {ENGINEERED_FEATURES_EW}")
    else:
        print(f"BASE_FEATURES: {BASE_FEATURES_NS}")
        print(f"ENGINEERED FEATURES: {ENGINEERED_FEATURES_NS}")
    print(f"FEATURES: {features}")


def main_CP(df_train: pd.DataFrame, df_test: pd.DataFrame, df_test_labels: pd.DataFrame, direction: str):
    # manually remove the change point at time index 0. We know that there is a time change, so we do not have to try
    # and predict it
    df_train.loc[df_train["TimeIndex"] == 0, "EW"] = 0
    df_train.loc[df_train["TimeIndex"] == 0, "NS"] = 0
    df_test.loc[df_test["TimeIndex"] == 0, "EW"] = 0
    df_test.loc[df_test["TimeIndex"] == 0, "NS"] = 0

    num_pos = df_train[direction].sum()
    all_samples = df_train[direction].shape[0]
    num_neg = all_samples - num_pos
    scale_pos = num_neg / num_pos
    rf = XGBClassifier(random_state=RANDOM_STATE, n_estimators=300, max_leaves=0, learning_rate=0.3, max_bin=256 * 8,
                       verbosity=2, tree_method="hist", scale_pos_weight=scale_pos, reg_lambda=0.5, max_depth=12)

    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW if direction == "EW" else BASE_FEATURES_NS
    # unwrap
    df_train[DEG_FEATURES] = np.unwrap(np.deg2rad(df_train[DEG_FEATURES]))
    df_test[DEG_FEATURES] = np.unwrap(np.deg2rad(df_test[DEG_FEATURES]))

    # FEATURE ENGINEERING
    feature_dict = ENGINEERED_FEATURES_EW if direction == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF, backfill to fill NANs resulting from window
            df_train[new_feature_name] = df_train.groupby(level=0, group_keys=False)[[feature]].apply(
                lambda_fnc).bfill()
            df_test[new_feature_name] = df_test.groupby(level=0, group_keys=False)[[feature]].apply(
                lambda_fnc).bfill()
            features.append(new_feature_name)

    # engineered_peaks and lag_features function both add feature name to list. to not add twice we have to make own
    # list for train and test
    features_train = features.copy()
    features_test = features.copy()

    # add engineered inclination feature
    df_train, features_train = add_engineered_peaks(df_train, features_train)
    df_test, features_test = add_engineered_peaks(df_test, features_test)

    # add lags
    df_train, features_train = add_lag_features(df_train, features_train, 8)
    df_test, features_test = add_lag_features(df_test, features_test, 8)

    # RF model
    print_params(features_train, direction)
    print("Fitting...")
    start_time = timer()
    # rf = load("../trained_model/state_classifier.joblib")
    rf.fit(df_train[features_train], df_train[direction])
    print(f"Took: {timer() - start_time:.3f} seconds")

    # Write classifier to disk
    model_name = f"state_classifier_{direction}.joblib"
    # Get the absolute directory path of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"../trained_model/{model_name}"
    file_path = os.path.join(base_dir, model_path)
    dump(rf, file_path, compress=3)
    print(f"Wrote model {model_name} to {file_path}")

    print("Predicting...")
    df_test["PREDICTED"] = rf.predict(df_test[features_test])

    # POST PROCESSING
    print("Postprocessing")
    df_test["PREDICTED_sum"] = df_test["PREDICTED"].rolling(4, center=True).sum()
    df_test["PREDICTED_CLEAN"] = 0
    df_test.loc[df_test["PREDICTED_sum"] >= 4, "PREDICTED_CLEAN"] = 1

    # removing two changepoints that are directly next to each other.
    diff = df_test["PREDICTED_CLEAN"].shift(1, fill_value=0) & df_test["PREDICTED_CLEAN"]
    df_test["PREDICTED_CLEAN"] -= diff

    # set start node manually
    df_test.loc[df_test["TimeIndex"] == 0, "PREDICTED_CLEAN"] = 1

    changepoints = df_test.loc[df_test["PREDICTED_CLEAN"].astype("bool"), ["ObjectID", "TimeIndex"]]
    changepoints["Direction"] = direction
    changepoints["Node"] = "EGAL"
    changepoints["Type"] = "EGAL"

    # EVALUATION
    object_ids = changepoints["ObjectID"].unique()
    labels = df_test_labels
    labels = labels.loc[labels["ObjectID"].isin(object_ids), :]
    lable_other_dir = labels[labels["Direction"] != direction]
    # Currently not predicting NS therefore just add the correct ones
    changepoints = pd.concat([changepoints, lable_other_dir])
    changepoints = changepoints.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    eval = NodeDetectionEvaluator(labels, changepoints, tolerance=6)
    precision, recall, f2, rmse = eval.score(debug=True)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {rmse:.2f}')

    oids = changepoints["ObjectID"].unique().tolist()
    wrong_ids = []
    wrong_FP = []
    wrong_FN = []
    for id in oids:
        tp, fp, fn, gt_object, p_object = eval.evaluate(id)
        if fp > 0:
            wrong_FP.append(id)
        if fn > 0:
            wrong_FN.append(id)
        if fp > 0 or fn > 0:
            wrong_ids.append(id)

    print(f"Num wrong time series: {len(wrong_ids)}")
    print(f"Num time series with FN: {len(wrong_FN)}")
    print(f"Num time series with FP: {len(wrong_FP)}")
    print(wrong_ids)

    return f2


TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_own")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_label_own.csv")

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
    ("std", lambda x: x.rolling(window=WINDOW_SIZE).std()):
        ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "Inclination (deg)"],  # , "RAAN (deg)"

}
ENGINEERED_FEATURES_NS = {
    ("std", lambda x: x.rolling(window=WINDOW_SIZE).std()):
        ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "Inclination (deg)"],
}

DEG_FEATURES = [
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Periapsis (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)"
]

WINDOW_SIZE = 6
TRAIN_TEST_RATIO = 0.8
RANDOM_STATE = 42
DIRECTION = "EW"
NUM_CSV_SETS = -1

if __name__ == "__main__":
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)

    object_ids = df['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids,
                                           test_size=1 - TRAIN_TEST_RATIO,
                                           random_state=RANDOM_STATE,
                                           shuffle=True)

    train_set = df.loc[train_ids].copy()
    test_set = df.loc[test_ids].copy()
    test_labels: pd.DataFrame = pd.read_csv(TRAIN_LABEL_PATH)
    test_labels = test_labels.loc[test_labels["ObjectID"].isin(test_ids), :]

    main_CP(train_set, test_set, test_labels, DIRECTION)
