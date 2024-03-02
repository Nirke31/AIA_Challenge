from pathlib import Path
from typing import Tuple, List

import torch
import lightning as L
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import load

from myModel import LitClassifier
from dataset_manip import SubmissionWindowDataset

import pandas as pd

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TRAINED_MODEL_DIR = "./trained_model/"
TEST_DATA_DIR = "./dataset/test/"
TEST_PREDS_FP = "./submission/submission.csv"

RF_BASE_FEATURES_EW = [
    "TBD"
]

RF_BASE_FEATURES_NS = [
    "TBD"
]

CLASSIFIER_FEATURES_EW = [
    "TBD"
]

CLASSIFIER_FEATURES_NS = [
    "TBD"
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
        ("kurt", lambda x: x.rolling(window=window_size).kurt()):
            ["Eccentricity", "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"],
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
    # Check if test_data is empty
    if not test_data_path:
        raise ValueError(f'No csv files found in {filepath}')

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

    return data, rf_features_ew, rf_features_ns


def generate_output(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


def main():
    # Load models for prediction
    rf_ew = load(TRAINED_MODEL_DIR + "state_classifier_EW.joblib")
    rf_ns = load(TRAINED_MODEL_DIR + "state_classifier_NS.joblib")
    classifier_ew: LitClassifier = LitClassifier.load_from_checkpoint(TRAINED_MODEL_DIR + "classification_EW.ckpt",
                                                                      sequence_len=11, feature_size=15, num_classes=5)
    classifier_ns: LitClassifier = LitClassifier.load_from_checkpoint(TRAINED_MODEL_DIR + "classification_NS.ckpt",
                                                                      sequence_len=11, feature_size=15, num_classes=5)
    # Read test dataset.
    df, rf_features_ew, rf_features_ns = load_test_data_and_preprocess(Path(TEST_DATA_DIR))

    # predict state change
    print("Predicting...")
    df["PREDICTED_EW"] = rf_ew.predict(df[rf_features_ew])
    df["PREDICTED_NS"] = rf_ns.predict(df[rf_features_ns])

    # post-processing / pre-processing for classification
    # Manually set the state change at timeindex 0
    df.loc[df.loc[:, "TimeIndex"] == 0, "PREDICTED_EW"] = 1
    df.loc[df.loc[:, "TimeIndex"] == 0, "PREDICTED_NS"] = 1
    # load datasets for classification
    scale_features = CLASSIFIER_FEATURES_EW + CLASSIFIER_FEATURES_NS
    df.loc[:, scale_features] = pd.DataFrame(StandardScaler().fit_transform(df.loc[:, scale_features]),
                                             index=df.index, columns=scale_features)

    ds_ew = SubmissionWindowDataset(df, CLASSIFIER_FEATURES_EW, "EW", window_size=11)
    ds_ns = SubmissionWindowDataset(df, CLASSIFIER_FEATURES_EW, "NS", window_size=11)
    dataloader_ew = DataLoader(ds_ew, batch_size=20, shuffle=True, num_workers=2)
    dataloader_ns = DataLoader(ds_ns, batch_size=10, shuffle=True, num_workers=2)

    # classification of type
    trainer = L.Trainer()
    test = trainer.predict(classifier_ew, dataloader_ew, return_predictions=True)
    test1 = trainer.predict(classifier_ns, dataloader_ns, return_predictions=True)

    # deduce nodes

    # store?


if __name__ == "__main__":
    main()
