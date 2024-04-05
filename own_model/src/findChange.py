import math
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


def print_params():
    print("PARAMS:")
    print(f"NUM CSVs: {NUM_CSV_SETS}")
    print(f"DIRECTION: {DIRECTION}")
    if DIRECTION == "EW":
        print(f"BASE_FEATURES: {BASE_FEATURES_EW}")
        print(f"ENGINEERED FEATURES: {ENGINEERED_FEATURES_EW}")
    else:
        print(f"BASE_FEATURES: {BASE_FEATURES_NS}")
        print(f"ENGINEERED FEATURES: {ENGINEERED_FEATURES_NS}")
    print(f"FEATURES: {features}")


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
    # df.to_pickle("../../dataset/df.pkl")
    # df: pd.DataFrame = pd.read_pickle("../../dataset/df.pkl")

    # manually remove the change point at time index 0. We know that there is a time change, so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0
    # RF approach
    print("Dataset loaded")
    object_ids = df['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids,
                                           test_size=1 - TRAIN_TEST_RATIO,
                                           random_state=RANDOM_STATE,
                                           shuffle=True)

    # rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=15,
    #                             class_weight="balanced_subsample", criterion="log_loss")
    # rf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced", learning_rate=0.597289,
    #                                     max_iter=213, early_stopping=False, max_leaf_nodes=None,
    #                                     min_samples_leaf=25, l2_regularization=0.1)
    num_pos = df[DIRECTION].sum()
    all_samples = df[DIRECTION].shape[0]
    num_neg = all_samples - num_pos
    scale_pos = num_neg / num_pos

    # rf = XGBClassifier(random_state=RANDOM_STATE, n_estimators=300, max_leaves=0, learning_rate=0.3,
    #                    verbosity=2, tree_method="hist", scale_pos_weight=scale_pos, reg_lambda=1.5, max_depth=11
    rf = XGBClassifier(random_state=RANDOM_STATE, n_estimators=300, max_leaves=0, learning_rate=0.3, max_bin=256 * 8,
                       verbosity=2, tree_method="hist", scale_pos_weight=scale_pos, reg_lambda=0.5, max_depth=12)

    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW if DIRECTION == "EW" else BASE_FEATURES_NS
    # unwrap
    df[DEG_FEATURES] = np.unwrap(np.deg2rad(df[DEG_FEATURES]))

    # FEATURE ENGINEERING
    feature_dict = ENGINEERED_FEATURES_EW if DIRECTION == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF, backfill to fill NANs resulting from window
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()
            features.append(new_feature_name)

    # add engineered inclination feature
    df, features = add_engineered_peaks(df, features)

    # add lags
    df, features = add_lag_features(df, features, 8)
    #
    # # adding smoothing because of some FPs
    # new_feature_name = "Inclination (deg)" + "_" + "std"
    # df[new_feature_name] = df.groupby(level=0, group_keys=False)[["Inclination (deg)"]].apply(
    #     lambda x: x.rolling(window=WINDOW_SIZE).std())
    # df[new_feature_name + "smoothed_1"] = df.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x[::-1].ewm(span=100, adjust=True).sum()[::-1])
    # df[new_feature_name + "smoothed_2"] = df.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x.ewm(span=100, adjust=True).sum())
    # df.bfill(inplace=True)
    # df.ffill(inplace=True)
    # df[new_feature_name + "_smoothed"] = (df[new_feature_name + "smoothed_1"] +
    #                                       df[new_feature_name + "smoothed_2"]) / 2
    # # test features
    # test_features_name = "inc_std_sum"
    # df[test_features_name] = df.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x.rolling(window=170).max())
    # features.append(test_features_name)
    # df.bfill(inplace=True)
    # df.ffill(inplace=True)

    # MANIPULATION DONE. TIME FOR TRAINING
    test_data = df.loc[test_ids].copy()
    train_data = df.loc[train_ids].copy()

    # RF model
    print_params()
    print("Fitting...")
    start_time = timer()
    # rf = load("../trained_model/state_classifier.joblib")
    rf.fit(train_data[features], train_data[DIRECTION])
    print(f"Took: {timer() - start_time:.3f} seconds")
    # Write classifier to disk
    dump(rf, "../trained_model/state_classifier.joblib", compress=3)

    # calibrated_clf = CalibratedClassifierCV(estimator=rf, method='isotonic')
    # calibrated_clf.fit(train_data[features], train_data[DIRECTION])

    print("Predicting...")
    # train_data["PREDICTED"] = rf.predict(train_data[features])
    test_data["PREDICTED"] = rf.predict(test_data[features])
    predict_proba: np.ndarray = rf.predict_proba(test_data[features])[:, 1]

    # Step 3: Calculate precision, recall for various thresholds
    precision, recall, thresholds = precision_recall_curve(test_data[DIRECTION], predict_proba)
    test = PrecisionRecallDisplay(precision, recall)
    # test.plot()
    # plt.show()

    # # Calculate F2 scores for each threshold
    # beta = 3
    # f2_scores = ((1 + beta*beta) * precision * recall) / (((beta*beta) * precision) + recall)
    # # Avoid division by zero
    # f2_scores = np.nan_to_num(f2_scores)
    #
    # # Step 4: Find the threshold that maximizes F2 score
    # max_f2_index = np.argmax(f2_scores)
    # # max_f2_index = 670603
    # best_threshold = thresholds[max_f2_index]
    # best_f2_score = f2_scores[max_f2_index]
    #
    # print(f"Best threshold: {best_threshold}")
    # print(f"Best F2 score: {best_f2_score}")
    # test_data["PREDICTED"] = (predict_proba >= best_threshold).astype('int')
    # POST PROCESSING
    print("Postprocessing")
    test_data["PREDICTED_sum"] = test_data["PREDICTED"].rolling(4, center=True).sum()
    test_data["PREDICTED_CLEAN"] = 0
    test_data.loc[test_data["PREDICTED_sum"] >= 4, "PREDICTED_CLEAN"] = 1

    # removing two changepoints that are directly next to each other.
    diff = test_data["PREDICTED_CLEAN"].shift(1, fill_value=0) & test_data["PREDICTED_CLEAN"]
    test_data["PREDICTED_CLEAN"] -= diff

    # set start node manually
    test_data.loc[test_data["TimeIndex"] == 0, "PREDICTED_CLEAN"] = 1

    changepoints = test_data.loc[test_data["PREDICTED_CLEAN"].astype("bool"), ["ObjectID", "TimeIndex"]]
    changepoints["Direction"] = DIRECTION
    changepoints["Node"] = "EGAL"
    changepoints["Type"] = "EGAL"

    # EVALUATION
    object_ids = changepoints["ObjectID"].unique()
    labels = pd.read_csv(TRAIN_LABEL_PATH)
    labels = labels.loc[labels["ObjectID"].isin(object_ids), :]
    lable_other_dir = labels[labels["Direction"] != DIRECTION]
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
    features_plot = [
        "Eccentricity",
        "Eccentricity_std",
        "Semimajor Axis (m)",
        "Inclination (deg)",
        "RAAN (deg)",
        # "Argument of Periapsis (deg)",
        # "True Anomaly (deg)",
        # "Latitude (deg)",
        # "Longitude (deg)",
        "smoothed",
        "peak",
        "peak_exp",
        "NS",
        "PREDICTED",
        "PREDICTED_sum",
        "PREDICTED_CLEAN"]
    for id in wrong_ids:
        test_data.loc[test_data["ObjectID"] == id, features_plot].plot(subplots=True, title=id)
        eval.plot(id)

    # eval.plot(157)  # object_ids[0]

    # print("TRAIN RESULTS:")
    # total_tp, total_fp, total_tn, total_fn = state_change_eval(torch.tensor(train_data["PREDICTED"].to_numpy()),
    #                                                            torch.tensor(train_data[DIRECTION].to_numpy()))
    # precision = total_tp / (total_tp + total_fp) \
    #     if (total_tp + total_fp) != 0 else 0
    # recall = total_tp / (total_tp + total_fn) \
    #     if (total_tp + total_fn) != 0 else 0
    # f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
    #     if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
    #
    # print(f"Total TPs: {total_tp}")
    # print(f"Total FPs: {total_fp}")
    # print(f"Total FNs: {total_fn}")
    # print(f"Total TNs: {total_tn}")
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F2: {f2:.2f}')
    #
    # print("TEST RESULTS:")
    # total_tp, total_fp, total_tn, total_fn = state_change_eval(torch.tensor(test_data["PREDICTED"].to_numpy()),
    #                                                            torch.tensor(test_data[DIRECTION].to_numpy()))
    # precision = total_tp / (total_tp + total_fp) \
    #     if (total_tp + total_fp) != 0 else 0
    # recall = total_tp / (total_tp + total_fn) \
    #     if (total_tp + total_fn) != 0 else 0
    # f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
    #     if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
    # print(f"Total TPs: {total_tp}")
    # print(f"Total FPs: {total_fp}")
    # print(f"Total FNs: {total_fn}")
    # print(f"Total TNs: {total_tn}")
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F2: {f2:.2f}')
    #
    # # feature importance with xgboost
    # feature_importances = rf.feature_importances_
    # xgb_importances = pd.Series(feature_importances, index=features)
    # xgb_importances.plot.bar()
    # plt.tight_layout()
    # plt.show()

    # # feature importance with permutation, more robust or smth like that
    # start_time = timer()
    # f2_scorer = make_scorer(fbeta_score, beta=2)
    # result = permutation_importance(rf, test_data[features], test_data[DIRECTION].to_numpy(), n_repeats=5,
    #                                 random_state=RANDOM_STATE, n_jobs=2, scoring=f2_scorer)
    # elapsed_time = timer() - start_time
    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    # forest_importances = pd.Series(result.importances_mean, index=features)
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()
    #
    # PrecisionRecallDisplay.from_estimator(rf, test_data[features], test_data[DIRECTION], name="RF",
    #                                       plot_chance_level=True)
    # plt.show()
