import math
import os
from pathlib import Path

import numpy as np
from joblib import dump

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L

from own_model.src.dataset_manip import load_data_window_ready, GetWindowDataset
from own_model.src.myModel import LitClassifier


def get_window_size(direction: str, first: bool) -> int:
    if first:
        if direction == "EW":
            return WINDOW_SIZE[2]
        else:  # NS
            return WINDOW_SIZE[3]
    else:
        if direction == "EW":
            return WINDOW_SIZE[0]
        else:  # NS
            return WINDOW_SIZE[1]


def main_CNN(train_data: pd.DataFrame, train_labels: pd.DataFrame, test_data: pd.DataFrame, test_labels: pd.DataFrame,
             direction: str, first: bool):
    L.seed_everything(RANDOM_STATE, workers=True)

    FEATURES = FEATURES_EW if direction == "EW" else FEATURES_NS
    # get window size.
    window_size = get_window_size(direction, first)

    # Train only first sample or without first sample
    if first:
        train_labels = train_labels[train_labels['TimeIndex'] == 0]
        test_labels = test_labels[test_labels['TimeIndex'] == 0]
    else:
        train_labels = train_labels[train_labels['TimeIndex'] != 0]
        test_labels = test_labels[test_labels['TimeIndex'] != 0]

    # unwrap
    train_data[DEG_FEATURES] = np.unwrap(np.deg2rad(train_data[DEG_FEATURES]))
    test_data[DEG_FEATURES] = np.unwrap(np.deg2rad(test_data[DEG_FEATURES]))

    # FEATURE ENGINEERING
    feature_dict = ENGINEERED_FEATURES_EW if direction == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF, backfill to fill NANs resulting from window
            train_data[new_feature_name] = train_data.groupby(level=0, group_keys=False)[[feature]].apply(
                lambda_fnc).bfill()
            test_data[new_feature_name] = test_data.groupby(level=0, group_keys=False)[[feature]].apply(
                lambda_fnc).bfill()
            FEATURES.append(new_feature_name)

    train_data = train_data[FEATURES]
    test_data = test_data[FEATURES]

    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)

    # Get the absolute directory path of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"../trained_model/scaler.joblib"
    file_path = os.path.join(base_dir, model_path)
    # this single dump of scaler currently only works because NS and EW have some features
    dump(scaler, file_path, compress=0)

    EW_labels = train_labels[train_labels["Direction"] == "EW"]
    NS_labels = train_labels[train_labels["Direction"] == "NS"]
    # shuffle targets
    labels = EW_labels.copy() if direction == "EW" else NS_labels.copy()  # sets here which direction we are training
    labels = labels.sample(frac=1)
    labels.reset_index(drop=True, inplace=True)
    split_train_test = math.floor(labels.shape[0] * TRAIN_VAL_RATIO)
    train_labels = labels.iloc[:split_train_test]
    val_labels = labels.iloc[split_train_test:]

    # reset index. needed for window dataset as it iterates over them via range(start, end) indices
    train_labels.reset_index(drop=True, inplace=True)
    val_labels.reset_index(drop=True, inplace=True)
    test_labels.reset_index(drop=True, inplace=True)

    ds_train = GetWindowDataset(train_data, train_labels, window_size)
    ds_val = GetWindowDataset(train_data, val_labels, window_size)
    ds_test = GetWindowDataset(test_data, test_labels, window_size)
    # check train vs val ratio
    print(f"Train label counts: {ds_train.tgt.loc[:, 'Type'].value_counts()}")
    print(f"Val label counts {ds_val.tgt.loc[:, 'Type'].value_counts()}")

    # sampler
    num_samples = ds_train.tgt.shape[0]
    labels_count: pd.Series = ds_train.tgt.loc[:, 'Type'].value_counts()
    class_counts = {key: value for key, value in labels_count.items()}
    # each sample is assigned a weight based on its class
    weights = [1 / class_counts[i] for i in ds_train.tgt.loc[:, 'Type'].to_list()]
    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)

    dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                  persistent_workers=True, pin_memory=True, sampler=sampler)
    dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                persistent_workers=True, pin_memory=True)
    dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS)

    print("Start fitting...")
    # Get the absolute directory path of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if first:
        model_name = f"classification_first_{direction}"
    else:
        model_name = f"classification_{direction}"
    model_path = f"../trained_model/"
    file_path = os.path.join(base_dir, model_path)
    complete_path = os.path.join(file_path, f"{model_name}.ckpt")
    # remove the last checkpoint. ModelCheckpoint cannot ovewrite :)))
    Path(complete_path).unlink(missing_ok=True)

    # get actual model
    # have to (re)set size because features were added
    SRC_SIZE = len(FEATURES)
    model = LitClassifier(window_size, SRC_SIZE, 1e-4, TGT_SIZE)
    early_stop_callback = EarlyStopping(monitor="val_MulticlassFBetaScore", mode="max", patience=40)
    checkpoint_callback = ModelCheckpoint(monitor="val_MulticlassFBetaScore",
                                          mode="max",
                                          dirpath=file_path,
                                          filename=model_name)
    trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        check_val_every_n_epoch=1)  # , accumulate_grad_batches=3

    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    scores = trainer.test(dataloaders=dataloader_test, ckpt_path="best")

    return scores[0]['test_MulticlassFBetaScore']


SHUFFLE_DATA = False
FEATURES_NS = [
    "Eccentricity",
    "Semimajor Axis (m)",
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Periapsis (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)",
    "Altitude (m)"
    # "X (m)",
    # "Y (m)",
    # "Z (m)",
    # "Vx (m/s)",
    # "Vy (m/s)",
    # "Vz (m/s)"
]
FEATURES_EW = [
    "Eccentricity",
    "Semimajor Axis (m)",
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Periapsis (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)",
    "Altitude (m)"
]

ENGINEERED_FEATURES_EW = {
    ("std", lambda x: x.rolling(window=6).std()):
        ["Eccentricity",
         "Semimajor Axis (m)",
         "Inclination (deg)",
         "RAAN (deg)",
         "Argument of Periapsis (deg)",
         "True Anomaly (deg)",
         "Latitude (deg)",
         "Longitude (deg)",
         "Altitude (m)"],
}
ENGINEERED_FEATURES_NS = {
    ("std", lambda x: x.rolling(window=6).std()):
        ["Eccentricity",
         "Semimajor Axis (m)",
         "Inclination (deg)",
         "RAAN (deg)",
         "Argument of Periapsis (deg)",
         "True Anomaly (deg)",
         "Latitude (deg)",
         "Longitude (deg)",
         "Altitude (m)"],
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

# User settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = False
LOAD_EVAL = False
RANDOM_STATE = 42
TRAINED_MODEL_NAME = "model.pkl"
TRAINED_MODEL_PATH = Path('./trained_model/' + TRAINED_MODEL_NAME)
TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_own/")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_label_own.csv")

TGT_SIZE = 5  # based on the dataset dict
TRAIN_TEST_RATIO = 0.8
TRAIN_VAL_RATIO = 0.8
BATCH_SIZE = 20
WINDOW_SIZE = [51, 101, 2101, 2101]  # EW, NS, EW first, NS first
EPOCHS = 400
DIRECTION = "NS"
FIRST = True
OWN_TEST_SET = False
NUM_WORKERS = 2
NUM_CSV_SETS = 100

if __name__ == "__main__":
    # FOR FITTING WINDOW MODEL
    data_train, labels_train = load_data_window_ready(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, NUM_CSV_SETS)

    if OWN_TEST_SET:
        data_test, labels_test = load_data_window_ready(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, NUM_CSV_SETS)
    else:
        object_ids = data_train['ObjectID'].unique()
        train_ids, test_ids = train_test_split(object_ids, test_size=1 - TRAIN_TEST_RATIO,
                                               random_state=RANDOM_STATE, shuffle=True)
        data = data_train.loc[data_train["ObjectID"].isin(train_ids), :].copy()
        data_test = data_train.loc[data_train["ObjectID"].isin(test_ids), :].copy()
        # had to temporary store it to not overwrite
        data_train = data

        labels = labels_train.loc[labels_train["ObjectID"].isin(train_ids), :]
        labels_test = labels_train.loc[labels_train["ObjectID"].isin(test_ids), :]
        labels_train = labels

    main_CNN(data_train, labels_train, data_test, labels_test, DIRECTION, FIRST)
