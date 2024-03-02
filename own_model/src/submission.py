from pathlib import Path
from typing import Tuple, List
import time

import torch
import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import load

from myModel import LitClassifier
from dataset_manip import SubmissionWindowDataset
from baseline_submissions.evaluation import NodeDetectionEvaluator

import pandas as pd

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TRAINED_MODEL_DIR = "../trained_model/"
TEST_DATA_DIR = "../../dataset/test/"
TEST_PREDS_FP = "../../submission/submission.csv"
DEBUG = True

RF_BASE_FEATURES_EW = ["Eccentricity",
                       "Semimajor Axis (m)",
                       "Inclination (deg)",
                       "RAAN (deg)",
                       "Argument of Periapsis (deg)",
                       "True Anomaly (deg)",
                       "Latitude (deg)",
                       "Longitude (deg)",
                       "Altitude (m)",
                       ]

RF_BASE_FEATURES_NS = ["Eccentricity",
                       "Semimajor Axis (m)",
                       "Inclination (deg)",
                       "RAAN (deg)",
                       "Argument of Periapsis (deg)",
                       "True Anomaly (deg)",
                       "Latitude (deg)",
                       "Longitude (deg)",
                       "Altitude (m)",
                       "X (m)",
                       "Y (m)",
                       "Z (m)",
                       "Vx (m/s)",
                       "Vy (m/s)",
                       "Vz (m/s)"
                       ]

CLASSIFIER_FEATURES_EW = ["Eccentricity",
                          "Semimajor Axis (m)",
                          "Inclination (deg)",
                          "RAAN (deg)",
                          "Argument of Periapsis (deg)",
                          "True Anomaly (deg)",
                          "Latitude (deg)",
                          "Longitude (deg)",
                          "Altitude (m)",
                          "X (m)",
                          "Y (m)",
                          "Z (m)",
                          "Vx (m/s)",
                          "Vy (m/s)",
                          "Vz (m/s)"
                          ]

CLASSIFIER_FEATURES_NS = ["Eccentricity",
                          "Semimajor Axis (m)",
                          "Inclination (deg)",
                          "RAAN (deg)",
                          "Argument of Periapsis (deg)",
                          "True Anomaly (deg)",
                          "Latitude (deg)",
                          "Longitude (deg)",
                          "Altitude (m)",
                          "X (m)",
                          "Y (m)",
                          "Z (m)",
                          "Vx (m/s)",
                          "Vy (m/s)",
                          "Vz (m/s)"
                          ]


def add_EW_features(data: pd.DataFrame, features_in: List[str], window_size: int) -> Tuple[pd.DataFrame, List[str]]:
    features_ew = features_in.copy()
    # features selected based on rf feature importance
    engineered_features = {
        ("var", lambda x: x.rolling(window=window_size).var()):
            ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)"],  # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
        ("skew", lambda x: x.rolling(window=window_size).skew()):
            ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
        ("kurt", lambda x: x.rolling(window=window_size).kurt()):
            ["Eccentricity", "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"],
        ("sem", lambda x: x.rolling(window=window_size).sem()):
            ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
    }

    # FEATURE ENGINEERING
    for (math_type, lambda_fnc), feature_list in engineered_features.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_EW"
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            data[new_feature_name] = data.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc)
            features_ew.append(new_feature_name)
    return data, features_ew


def add_NS_features(data: pd.DataFrame, features_in: List[str], window_size: int) -> Tuple[pd.DataFrame, List[str]]:
    features_ns = features_in.copy()
    # features selected based on rf feature importance
    engineered_features = {
        ("var", lambda x: x.rolling(window=window_size).var()):
            ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)"],  # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
        ("skew", lambda x: x.rolling(window=window_size).skew()):
            ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
        # ("kurt", lambda x: x.rolling(window=window_size).kurt()):
        #     ["Eccentricity", "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"],
        ("sem", lambda x: x.rolling(window=window_size).sem()):
            ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
    }

    # FEATURE ENGINEERING
    for (math_type, lambda_fnc), feature_list in engineered_features.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_NS"
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            data[new_feature_name] = data.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc)
            features_ns.append(new_feature_name)
    return data, features_ns


def load_test_data_and_preprocess(filepath: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    test_data_path = Path(filepath).glob('*.csv')

    # load data
    merged_data = pd.DataFrame()
    for data_file in test_data_path:
        data_df = pd.read_csv(data_file)
        data_df['ObjectID'] = int(data_file.stem)
        data_df['TimeIndex'] = range(len(data_df))
        merged_data = pd.concat([merged_data, data_df])

    md_index = pd.MultiIndex.from_frame(merged_data[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
    merged_data.index = md_index
    merged_data.sort_index(inplace=True)

    # loaded, now create EW and NS dataframe and features
    data, rf_features_ew = add_EW_features(merged_data, RF_BASE_FEATURES_EW, 6)
    data, rf_features_ns = add_NS_features(data, RF_BASE_FEATURES_NS, 3)
    data = data.bfill()

    return data, rf_features_ew, rf_features_ns


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
        if data != "NK":
            prev_type = data
            continue
        # we are not station keeping, deduce if Node is ID or AD
        if prev_type in SK:
            # we were station keeping, so now we have ID
            df_ew.loc[idx, "Node"] = "ID"
        elif prev_type in ["ID", "AD"]:
            # already not station keeping -> AD
            df_ew.loc[idx, "Node"] = "AD"

        prev_type = data

    # NS TRANSLATION
    station_keeping_nodes = df_ns.loc[:, "Type"].isin(SK)
    df_ns.loc[station_keeping_nodes, "Node"] = "IK"
    df_ns.loc[df_ns.loc[:, "Type"] == "NK", "Node"] = "IK"

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
    # Load models for prediction
    rf_ew = load(TRAINED_MODEL_DIR + "state_classifier_EW.joblib")
    rf_ns = load(TRAINED_MODEL_DIR + "state_classifier_NS.joblib")
    classifier_ew: LitClassifier = LitClassifier.load_from_checkpoint(TRAINED_MODEL_DIR + "classification_EW.ckpt")
    classifier_ns: LitClassifier = LitClassifier.load_from_checkpoint(TRAINED_MODEL_DIR + "classification_NS.ckpt")
    # Read test dataset.
    df, rf_features_ew, rf_features_ns = load_test_data_and_preprocess(Path(TEST_DATA_DIR))

    # predict state change
    df["PREDICTED_EW"] = rf_ew.predict(df[rf_features_ew])
    df["PREDICTED_NS"] = rf_ns.predict(df[rf_features_ns])

    # post-processing / pre-processing for classification
    # Manually set the state change at timeindex 0
    df.loc[df.loc[:, "TimeIndex"] == 0, "PREDICTED_EW"] = 1
    df.loc[df.loc[:, "TimeIndex"] == 0, "PREDICTED_NS"] = 1
    # load datasets for classification
    # get unique vals because in both are base features
    scale_features = list(set(CLASSIFIER_FEATURES_EW + CLASSIFIER_FEATURES_NS))
    df.loc[:, scale_features] = pd.DataFrame(StandardScaler().fit_transform(df.loc[:, scale_features]),
                                             index=df.index, columns=scale_features)

    ds_ew = SubmissionWindowDataset(df, CLASSIFIER_FEATURES_EW, "EW", window_size=11)
    ds_ns = SubmissionWindowDataset(df, CLASSIFIER_FEATURES_EW, "NS", window_size=11)
    dataloader_ew = DataLoader(ds_ew, batch_size=20, num_workers=2)
    dataloader_ns = DataLoader(ds_ns, batch_size=10, num_workers=2)

    # classification of type
    trainer = L.Trainer()
    prediction_list_ew = trainer.predict(classifier_ew, dataloader_ew, return_predictions=True)
    prediction_list_ns = trainer.predict(classifier_ns, dataloader_ns, return_predictions=True)
    # output is a list of tuples. Refactor to tensor
    prediction_list_ew = torch.cat([torch.stack([a, b, c], dim=1) for a, b, c in prediction_list_ew], dim=0)
    prediction_list_ns = torch.cat([torch.stack([a, b, c], dim=1) for a, b, c in prediction_list_ns], dim=0)
    # shape is (50, 3) but I want (3, 50)
    # prediction_list_ew = prediction_list_ew.permute(1, 0)
    # prediction_list_ns = prediction_list_ns.permute(1, 0)

    # deduce nodes
    test_results = generate_output(prediction_list_ew, prediction_list_ns, ds_ew.tgt_dict_int_to_str)

    # Save the test results to a csv file to be submitted to the challenge
    test_results.to_csv(TEST_PREDS_FP, index=False)
    print("Saved predictions to: {}".format(TEST_PREDS_FP))
    if DEBUG:
        time.sleep(360)  # TEMPORARY FIX TO OVERCOME EVALAI BUG


if __name__ == "__main__":
    main()
    if DEBUG:
        ground_truth = pd.read_csv("../../dataset/phase_1_v2/train_labels.csv")
        own = pd.read_csv("../../submission/submission.csv")
        NodeDetectionEvaluator(ground_truth, own, tolerance=6)
