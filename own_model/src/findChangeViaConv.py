import math
import pickle
import time
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor
import lightning as L
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from own_model.src.dataset_manip import load_data, state_change_eval, MyDataset, ChangePointDataset, split_train_test
from own_model.src.myModel import LitChangePointClassifier


# https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

def main():
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0.0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0.0

    # features, leaves room for potential feature engineering
    features = BASE_FEATURES_EW if DIRECTION == "EW" else BASE_FEATURES_NS
    print("Dataset loaded")

    # feature engineering
    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW if DIRECTION == "EW" else BASE_FEATURES_NS
    engineered_features_ew = {
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)"],
        ("kurt", lambda x: x.rolling(window=window_size).kurt()):
            ["Eccentricity"],
    }
    engineered_features_ns = {
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
    window_size = 6
    feature_dict = engineered_features_ew if DIRECTION == "EW" else engineered_features_ns
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_" + DIRECTION
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc)
            features.append(new_feature_name)

    # for feature in FEATURES:
    #     df[feature + "_var"] = df.groupby(level=0, group_keys=False)[[feature]].apply(
    #         lambda x: x.rolling(window=window_size).var())
    #
    # FEATURES = FEATURES + [x + "_var" for x in FEATURES]

    # fill beginning of rolling window (NaNs). Shouldn't really matter anyways? Maybe else Median
    df = df.bfill()

    # splitting. This splitting has the big problem that I cannot really control how many positive samples are
    # in train and test set. If I am unlucky, all are in one of the two.
    df_train, df_test = split_train_test(df, TRAIN_TEST_RATIO, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    df_train.loc[:, features] = pd.DataFrame(scaler.fit_transform(df_train.loc[:, features]),
                                             index=df_train.index, columns=features)
    df_test.loc[:, features] = pd.DataFrame(scaler.transform(df_test.loc[:, features]),
                                            index=df_test.index, columns=features)

    # load dataset and dataloader
    start_time = time.perf_counter()
    ds_train = ChangePointDataset(df_train.loc[:, features], df_train.loc[:, DIRECTION], WINDOW_SIZE)
    ds_test = ChangePointDataset(df_test.loc[:, features], df_test.loc[:, DIRECTION], WINDOW_SIZE)
    print(f"Time: {time.perf_counter() - start_time:4.0f}sec - for dataset to load")
    ratio_pos_neg_train = sum(ds_train.tgt) / (ds_train.tgt.shape[0] - sum(ds_train.tgt))
    ratio_pos_neg_test = sum(ds_test.tgt) / (ds_test.tgt.shape[0] - sum(ds_test.tgt))
    print(f"Ratio pos to neg in train: {ratio_pos_neg_train}")
    print(f"Ratio pos to neg in test: {ratio_pos_neg_test}")

    # Create random sampler to sample more positives than negatives
    num_samples = ds_train.tgt.shape[0]
    num_pos = ds_train.tgt.sum()
    num_neg = num_samples - ds_train.tgt.sum()
    class_counts = {1: num_pos, 0: num_neg // 4}  # batch has 1/4 pos samples and 3/4 neg.
    # each sample is assigned a weight based on its class
    weights = [1 / class_counts[i] for i in ds_train.tgt]
    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)

    dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                  persistent_workers=True, pin_memory=True, sampler=sampler)
    # For val (and test) we do not need a sampler because we just look at all values once
    dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                persistent_workers=True, pin_memory=True)
    dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA,
                                 num_workers=NUM_WORKERS)  # bit of leftover

    print("Start fitting...")

    # get actual model
    early_stop_callback = EarlyStopping(monitor="val_BinaryFBetaScore", mode="max", patience=3)
    trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                        callbacks=[early_stop_callback],
                        check_val_every_n_epoch=1)  # , accumulate_grad_batches=5
    model = LitChangePointClassifier(len(features), WINDOW_SIZE)
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model=model, dataloaders=dataloader_test)

    # store model
    trainer.save_checkpoint("../trained_model/changepoint.ckpt")

    return


if __name__ == "__main__":
    TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
    TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")

    BASE_FEATURES_EW = [
        # "Eccentricity",
        # "Semimajor Axis (m)",
        "Inclination (deg)",
        "RAAN (deg)",
        # "Argument of Periapsis (deg)",
        # "True Anomaly (deg)",
        # "Latitude (deg)",
        # "Longitude (deg)",
        # "Altitude (m)",
    ]

    # inclination, raan, std semimajor

    BASE_FEATURES_NS = ["Eccentricity",
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

    TRAIN_TEST_RATIO = 0.8
    RANDOM_STATE = 42
    NUM_WORKERS = 4
    EPOCHS = 100
    BATCH_SIZE = 1024
    SHUFFLE_DATA = False  # no real effect with WeightedRandomSampler
    WINDOW_SIZE = 51
    DIRECTION = "EW"
    NUM_CSV_SETS = -1

    L.seed_everything(RANDOM_STATE, workers=True)
    main()
