from pathlib import Path
from typing import Tuple, List
import time

import torch
import lightning as L
from scipy.signal import savgol_filter, find_peaks
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd
import numpy as np

from myModel import LitClassifier
from dataset_manip import SubmissionWindowDataset

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
DEBUG = False

if DEBUG:
    TRAINED_MODEL_DIR = "../trained_model/"
    TEST_DATA_DIR = "../../dataset/phase_1_v3/train/"
    TEST_PREDS_FP = "../../submission/submission.csv"
else:
    TRAINED_MODEL_DIR = "/trained_model/"
    TEST_DATA_DIR = "/dataset/test/"
    TEST_PREDS_FP = "/submission/submission.csv"

WINDOW_SIZE = 6

RF_BASE_FEATURES_EW = [
    "Longitude (deg)",
]
RF_BASE_FEATURES_NS = [
    "Eccentricity",
    "Inclination (deg)",
    "RAAN (deg)",
    "Longitude (deg)",
]

ENGINEERED_FEATURES_EW = {
    ("std", lambda x: x.rolling(window=WINDOW_SIZE).std()):
        ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "Inclination (deg)"]

}
ENGINEERED_FEATURES_NS = {
    ("std", lambda x: x.rolling(window=WINDOW_SIZE).std()):
        ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "Inclination (deg)"]
}

CLASSIFIER_FEATURES = ["Eccentricity",
                       "Semimajor Axis (m)",
                       "Inclination (deg)",
                       "RAAN (deg)",
                       "Argument of Periapsis (deg)",
                       "True Anomaly (deg)",
                       "Latitude (deg)",
                       "Longitude (deg)",
                       "Altitude (m)",
                       # "X (m)",
                       # "Y (m)",
                       # "Z (m)",
                       # "Vx (m/s)",
                       # "Vy (m/s)",
                       # "Vz (m/s)"
                       ]
ENGINEERED_FEATURES_CLASSIFIER = {
    ("std", lambda x: x.rolling(window=WINDOW_SIZE).std()):
        ["Eccentricity", "Semimajor Axis (m)", "Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)",
         "True Anomaly (deg)", "Latitude (deg)", "Longitude (deg)", "Altitude (m)"]
}

DEG_FEATURES = [
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Periapsis (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)"
]


def get_peaks(x: pd.DataFrame):
    objectID = x.index.get_level_values(0).unique()
    x["smoothed"] = savgol_filter(x["Inclination (deg)_std"], 13, 3)

    inc_std_average = x["smoothed"].median()

    peaks, _ = find_peaks(x["smoothed"], height=inc_std_average * 4)

    x.loc[(objectID, peaks), "peak"] = 1
    return x


def dynamic_weighting(x: pd.DataFrame):
    index = np.arange(x.shape[0])
    peak_index: np.array = index[x["peak"] == 1]

    # if we have 0 or just one peak we cannot calculate a period
    if peak_index.size > 1:
        # Calculate distance between consecutive peaks
        distance = np.diff(peak_index)
        # Determine expected period, may be a bit broke if we only have a few distances
        period = int(np.median(distance))

        x["peak_exp"] = x["peak"][::-1].rolling(window=period).max().bfill()[::-1]
        x["peak_exp"] = x["peak_exp"].ewm(span=period).sum()
    return x


def add_engineered_peaks(data: pd.DataFrame) -> pd.DataFrame:
    # already allocating features space
    data["peak"] = 0.0
    data["peak_exp"] = 0.0
    data["smoothed"] = 0.0
    # get peaks
    data = data.groupby(level=0, group_keys=False).apply(get_peaks)
    # weight peaks
    data = data.groupby(level=0, group_keys=False).apply(dynamic_weighting)

    return data


def add_engineered_features(data: pd.DataFrame, features_in: List[str], engineered_features: dict) \
        -> Tuple[pd.DataFrame, List[str]]:
    existing_features = data.columns.tolist()
    features = features_in.copy()

    # type and some kind of lambda fnc (std, mean, etc.)
    for (math_type, lambda_fnc), feature_list in engineered_features.items():
        # add for all features that should get math_type
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type
            # only add if not already existing
            if new_feature_name not in existing_features:
                # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
                # and then added back to the DF
                data[new_feature_name] = data.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()

            # always append to features
            features.append(new_feature_name)

    return data, features


def add_lag_features(df: pd.DataFrame, feature_cols: List[str], lag_steps: int):
    new_columns = pd.DataFrame({f"{col}_lag{i}": df.groupby(level=0, group_keys=False)[col].shift(i * 3)
                                for i in range(1, lag_steps + 1)
                                for col in feature_cols}, index=df.index)
    new_columns_neg = pd.DataFrame({f"{col}_lag-{i}": df.groupby(level=0, group_keys=False)[col].shift(i * -3)
                                    for i in range(1, lag_steps + 1)
                                    for col in feature_cols}, index=df.index)
    new_df = pd.concat([new_columns, new_columns_neg], axis=1)
    # basic features were maybe already added, therefore check if these coloumns already exist and don't add them
    new_df = new_df[new_df.columns.difference(df.columns)]
    df_out = pd.concat([df, new_df], axis=1)
    features_out = feature_cols + new_columns.columns.tolist() + new_columns_neg.columns.to_list()
    # fill nans
    df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.bfill())
    df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.ffill())
    return df_out, features_out


def add_EW_features(data: pd.DataFrame, features_in: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    features_ew = features_in.copy()
    # FEATURE ENGINEERING
    data, features_ew = add_engineered_features(data, features_ew, ENGINEERED_FEATURES_EW)

    return data, features_ew


def add_NS_features(data: pd.DataFrame, features_in: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    features_ns = features_in.copy()

    # FEATURE ENGINEERING
    data, features_ns = add_engineered_features(data, features_ns, ENGINEERED_FEATURES_NS)

    return data, features_ns


def load_test_data_and_preprocess(filepath: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    test_data_path = Path(filepath).glob('*.csv')

    # load data
    loaded_dfs = []
    for data_file in test_data_path:
        data_df = pd.read_csv(data_file)
        data_df['ObjectID'] = int(data_file.stem)
        data_df['TimeIndex'] = range(len(data_df))
        loaded_dfs.append(data_df)

    merged_data = pd.concat(loaded_dfs, axis=0)

    md_index = pd.MultiIndex.from_frame(merged_data[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
    merged_data.index = md_index
    merged_data.sort_index(inplace=True)
    # data loaded

    # unwrap
    merged_data[DEG_FEATURES] = np.unwrap(np.deg2rad(merged_data[DEG_FEATURES]))

    # create EW and NS dataframe and features
    data, rf_features_ew = add_EW_features(merged_data, RF_BASE_FEATURES_EW)
    data, rf_features_ns = add_NS_features(data, RF_BASE_FEATURES_NS)

    # already allocating features space
    data = add_engineered_peaks(data)
    rf_features_ew.append("peak_exp")
    rf_features_ns.append("peak_exp")

    # add lag
    data, rf_features_ew = add_lag_features(data, rf_features_ew, 8)
    data, rf_features_ns = add_lag_features(data, rf_features_ns, 8)

    # # adding smoothing because of some FPs
    # new_feature_name = "Inclination (deg)" + "_" + "std"
    # data[new_feature_name] = data.groupby(level=0, group_keys=False)[["Inclination (deg)"]].apply(
    #     lambda x: x.rolling(window=WINDOW_SIZE).std())
    # data[new_feature_name + "smoothed_1"] = data.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x[::-1].ewm(span=100, adjust=True).sum()[::-1])
    # data[new_feature_name + "smoothed_2"] = data.groupby(level=0, group_keys=False)[[new_feature_name]].apply(
    #     lambda x: x.ewm(span=100, adjust=True).sum())
    # data.bfill(inplace=True)
    # data.ffill(inplace=True)
    # data[new_feature_name + "_smoothed"] = (data[new_feature_name + "smoothed_1"] +
    #                                         data[new_feature_name + "smoothed_2"]) / 2
    # rf_features_ew.append(new_feature_name)
    # rf_features_ns.append(new_feature_name)
    # rf_features_ew.append(new_feature_name + "_smoothed")
    # rf_features_ns.append(new_feature_name + "_smoothed")

    return data, rf_features_ew, rf_features_ns


def classifier_preprocessing(df: pd.DataFrame, features_in: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    features = features_in.copy()
    # FEATURE ENGINEERING

    df, features = add_engineered_features(df, features, ENGINEERED_FEATURES_CLASSIFIER)

    # Manually set the state change at timeindex 0
    df["PREDICTED_FIRST_EW"] = 0
    df["PREDICTED_FIRST_NS"] = 0
    df.loc[df.loc[:, "TimeIndex"] == 0, "PREDICTED_FIRST_EW"] = 1
    df.loc[df.loc[:, "TimeIndex"] == 0, "PREDICTED_FIRST_NS"] = 1

    return df, features


def changepoint_postprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # post processing
    df["PREDICTED_SUM_EW"] = df["PREDICTED_EW"].rolling(5, center=True).sum()
    df["PREDICTED_SUM_NS"] = df["PREDICTED_NS"].rolling(3, center=True).sum()
    df["PREDICTED_CLEAN_EW"] = 0
    df["PREDICTED_CLEAN_NS"] = 0
    df.loc[df["PREDICTED_SUM_EW"] >= 5, "PREDICTED_CLEAN_EW"] = 1
    df.loc[df["PREDICTED_SUM_NS"] >= 3, "PREDICTED_CLEAN_NS"] = 1

    # removing two changepoints that are directly next to each other.
    diff = df["PREDICTED_CLEAN_EW"].shift(1, fill_value=0) & df["PREDICTED_CLEAN_EW"]
    df["PREDICTED_CLEAN_EW"] -= diff
    diff = df["PREDICTED_CLEAN_NS"].shift(1, fill_value=0) & df["PREDICTED_CLEAN_NS"]
    df["PREDICTED_CLEAN_NS"] -= diff

    return df


def generate_output(pred_ew: Tensor, pred_ns: Tensor, int_to_str_translation: dict) -> pd.DataFrame:
    """
        Translates predictions to final output dataframe. Only the type was predicted. We have to deduce Node.
        For all direction and types:
            if TimeIndex == 0:      -> SS
        For Direction == NS:
            if Type == EK|HK|Ck     -> IK
            if Type == NK:          -> ID
        For Direction == EW:
            if Type == EK|HK|Ck     -> IK
            if Type == NK:
                look at prev point
                if prev Type == EK|HK|Ck    -> ID
                if prev Type == ID|AD       -> AD
    Args:
        pred_ew: tensor of shape(3, num predictions_ew), order is prediction, objectID, timeIndex
        pred_ns: tensor of shape(3, num predictions_ns), order is prediction, objectID, timeIndex
        int_to_str_translation: dict from MyDataset

    Returns: pd.Dataframe, ready for comparison with NodeEvaluator
    """

    # Station keeping type is one of the following
    SK = ["EK", "HK", "CK"]
    df_ew = pd.DataFrame(pred_ew, columns=["Type", "ObjectID", "TimeIndex"], dtype="int64")
    df_ns = pd.DataFrame(pred_ns, columns=["Type", "ObjectID", "TimeIndex"], dtype="int64")
    df_ew.sort_values(by=["ObjectID", "TimeIndex"], inplace=True)
    df_ns.sort_values(by=["ObjectID", "TimeIndex"], inplace=True)

    df_ns["Direction"] = "NS"
    df_ew["Direction"] = "EW"
    # map integers back to string
    df_ew.loc[:, "Type"] = df_ew.loc[:, "Type"].map(lambda x: int_to_str_translation[x])
    df_ns.loc[:, "Type"] = df_ns.loc[:, "Type"].map(lambda x: int_to_str_translation[x])

    # create Nodes, first setting everything to NAN will result in a crash if following code does smth stupid
    df_ew["Node"] = "NAN"
    df_ns["Node"] = "NAN"

    # EW TRANSLATION
    station_keeping_nodes = df_ew.loc[:, "Type"].isin(SK)
    df_ew.loc[station_keeping_nodes, "Node"] = "IK"

    # we have to iterate over the dataframe because we have to know the previous Type
    # There is nothing I can do. It is what it is.
    prev_type = "NONE"
    for idx, data in df_ew.loc[:, "Type"].items():
        if data == "NK":
            # we are not station keeping, deduce if Node is ID or AD
            if prev_type in SK:
                # we were station keeping, so now we have ID
                df_ew.loc[idx, "Node"] = "ID"
            else:
                # already not station keeping -> AD
                df_ew.loc[idx, "Node"] = "AD"

        prev_type = data

    # NS TRANSLATION
    station_keeping_nodes = df_ns.loc[:, "Type"].isin(SK)
    df_ns.loc[station_keeping_nodes, "Node"] = "IK"
    df_ns.loc[df_ns.loc[:, "Type"] == "NK", "Node"] = "ID"

    # set beginning node. This (correctly) overwrite some Node labels set above
    df_ns.loc[df_ns.loc[:, "TimeIndex"] == 0, "Node"] = "SS"
    df_ew.loc[df_ew.loc[:, "TimeIndex"] == 0, "Node"] = "SS"

    # MERGE THE TWO
    results = pd.concat([df_ew, df_ns], axis=0, ignore_index=True)

    # sort for better (manual) comparison
    results_index = pd.MultiIndex.from_frame(results[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
    results.index = results_index
    results.sort_index(inplace=True)

    return results


def main():
    start_time = time.perf_counter()
    # Load models for prediction
    rf_ew = load(TRAINED_MODEL_DIR + "state_classifier_EW_xgboost.joblib")
    rf_ns = load(TRAINED_MODEL_DIR + "state_classifier_NS_xgboost.joblib")
    # rf models were trained with more than 4 cpus
    # update_rf_params = {"n_jobs": 4}
    # rf_ew = rf_ew.set_params(**update_rf_params)
    # rf_ns = rf_ns.set_params(**update_rf_params)
    classifier_ew: LitClassifier = LitClassifier.load_from_checkpoint(
        TRAINED_MODEL_DIR + "classification_epoch=60_val_MulticlassFBetaScore=0.97_EW_51_3c_2l.ckpt")
    classifier_ns: LitClassifier = LitClassifier.load_from_checkpoint(
        TRAINED_MODEL_DIR + "classification_epoch=37_val_MulticlassFBetaScore=0.99_NS_101_3c_2l.ckpt")
    classifier_first_ew: LitClassifier = LitClassifier.load_from_checkpoint(
        TRAINED_MODEL_DIR + "classification_epoch=67_val_MulticlassFBetaScore=0.95_EW_1501_3c_2l_first.ckpt")
    classifier_first_ns: LitClassifier = LitClassifier.load_from_checkpoint(
        TRAINED_MODEL_DIR + "classification_epoch=64_val_MulticlassFBetaScore=0.96_NS_2001_3c_2l_first.ckpt")
    # Load scaler for LitClassifier
    scaler: StandardScaler = load(TRAINED_MODEL_DIR + "scaler.joblib")

    # Read test dataset.
    df, rf_features_ew, rf_features_ns = load_test_data_and_preprocess(Path(TEST_DATA_DIR))
    print(f"Time: {time.perf_counter() - start_time:4.0f} sec - Dataset loaded")
    start_time = time.perf_counter()

    # PREDICT CHANGEPOINTS ---------------------------------------------------------------------------------------------

    # predict state change
    df["PREDICTED_EW"] = rf_ew.predict(df[rf_features_ew])
    df["PREDICTED_NS"] = rf_ns.predict(df[rf_features_ns])
    print(f"Time: {time.perf_counter() - start_time:4.0f} sec - States predicted")
    start_time = time.perf_counter()

    # post-processing
    df = changepoint_postprocessing(df)

    # CHANGEPOINTS DONE. NOW CLASSIFICAITON ----------------------------------------------------------------------------
    df, classifier_features = classifier_preprocessing(df, CLASSIFIER_FEATURES)
    # EW and NS have the same classification features
    df.loc[:, classifier_features] = scaler.transform(df.loc[:, classifier_features])

    ds_ew = SubmissionWindowDataset(df, classifier_features, "PREDICTED_CLEAN_EW", window_size=51)
    ds_ns = SubmissionWindowDataset(df, classifier_features, "PREDICTED_CLEAN_NS", window_size=101)
    ds_first_ew = SubmissionWindowDataset(df, classifier_features, "PREDICTED_FIRST_EW", window_size=1501)
    ds_first_ns = SubmissionWindowDataset(df, classifier_features, "PREDICTED_FIRST_NS", window_size=2001)
    dataloader_ew = DataLoader(ds_ew, batch_size=20, num_workers=1)
    dataloader_ns = DataLoader(ds_ns, batch_size=20, num_workers=1)
    dataloader_first_ew = DataLoader(ds_first_ew, batch_size=20, num_workers=1)
    dataloader_first_ns = DataLoader(ds_first_ns, batch_size=20, num_workers=1)
    print(f"Time: {time.perf_counter() - start_time:4.0f}sec - Dataloader")
    start_time = time.perf_counter()

    # classification of type
    trainer_ew = L.Trainer()
    trainer_ns = L.Trainer()
    trainer_first_ew = L.Trainer()
    trainer_first_ns = L.Trainer()
    prediction_list_ew = trainer_ew.predict(classifier_ew, dataloader_ew, return_predictions=True)
    prediction_list_ns = trainer_ns.predict(classifier_ns, dataloader_ns, return_predictions=True)
    prediction_list_first_ew = trainer_first_ew.predict(classifier_first_ew, dataloader_first_ew,
                                                        return_predictions=True)
    prediction_list_first_ns = trainer_first_ns.predict(classifier_first_ns, dataloader_first_ns,
                                                        return_predictions=True)
    print(f"Time: {time.perf_counter() - start_time:4.0f}sec - Type predicted")
    start_time = time.perf_counter()

    # CLASSIFICATION DONE. POSTPROCESSING ------------------------------------------------------------------------------

    # classification "postprocessing"
    # output is a list of tuples. Refactor to tensor
    prediction_list_ew = torch.cat([torch.stack([a, b, c], dim=1) for a, b, c in prediction_list_ew], dim=0)
    prediction_list_ns = torch.cat([torch.stack([a, b, c], dim=1) for a, b, c in prediction_list_ns], dim=0)
    prediction_list_first_ew = torch.cat([torch.stack([a, b, c], dim=1) for a, b, c in prediction_list_first_ew], dim=0)
    prediction_list_first_ns = torch.cat([torch.stack([a, b, c], dim=1) for a, b, c in prediction_list_first_ns], dim=0)
    # adding the first samples
    prediction_list_ew = torch.concat([prediction_list_ew, prediction_list_first_ew], dim=0)
    prediction_list_ns = torch.concat([prediction_list_ns, prediction_list_first_ns], dim=0)

    # deduce nodes
    test_results = generate_output(prediction_list_ew, prediction_list_ns, ds_ew.tgt_dict_int_to_str)
    print(f"Time: {time.perf_counter() - start_time:4.0f}sec - Output generated")

    # FINISHED. STORE RESULTS ------------------------------------------------------------------------------------------

    # Save the test results to a csv file to be submitted to the challenge
    test_results.to_csv(TEST_PREDS_FP, index=False)
    print("Saved predictions to: {}".format(TEST_PREDS_FP))

    if not DEBUG:
        print("Waiting for EVALAI...")
        time.sleep(360)  # TEMPORARY FIX TO OVERCOME EVALAI BUG


if __name__ == "__main__":
    main()
    if DEBUG:
        from baseline_submissions.evaluation import NodeDetectionEvaluator

        ground_truth = pd.read_csv("../../dataset/phase_1_v3/train_labels.csv")
        own = pd.read_csv("../../submission/submission.csv")
        test = NodeDetectionEvaluator(ground_truth, own, tolerance=6)
        precision, recall, f2, rmse = test.score(debug=True)
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2: {f2:.2f}')
        print(f'RMSE: {rmse:.2f}')
        test.plot(1151)
