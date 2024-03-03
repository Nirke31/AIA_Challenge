import math
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from own_model.src.dataset_manip import load_data, state_change_eval, MyDataset, ChangePointDataset
from own_model.src.myModel import LitChangePointClassifier

TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")

BASE_FEATURES_EW = ["Eccentricity",
                    "Semimajor Axis (m)",
                    "Inclination (deg)",
                    "RAAN (deg)",
                    "Argument of Periapsis (deg)",
                    "True Anomaly (deg)",
                    "Latitude (deg)",
                    "Longitude (deg)",
                    "Altitude (m)",
                    ]

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
EPOCHS = 500
BATCH_SIZE = 100
SHUFFLE_DATA = False
WINDOW_SIZE = 5
DIRECTION = "EW"
NUM_CSV_SETS = 1000

L.seed_everything(RANDOM_STATE, workers=True)


def main():
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0

    # features, leaves room for potential feature engineering
    features = BASE_FEATURES_EW if DIRECTION == "EW" else BASE_FEATURES_NS

    # label creation
    labels = df.loc[:, DIRECTION]
    # shuffle targets
    labels = labels.sample(frac=1)
    labels.reset_index(drop=True, inplace=True)
    print("Dataset loaded")

    # generate train/val/test data
    mask = np.random.rand(len(df)) < TRAIN_TEST_RATIO

    df_train = df[mask]
    df_test = df[~mask]
    label_train = labels[mask]
    label_test = labels[~mask]

    # Scale
    scaler = StandardScaler()
    df_train.loc[:, features] = pd.DataFrame(scaler.fit_transform(df_train.loc[:, features]),
                                             index=df_train.index, columns=features)
    df_test.loc[:, features] = pd.DataFrame(scaler.transform(df_test.loc[:, features]),
                                            index=df_test.index, columns=features)

    # load dataset and dataloader
    ds_train = ChangePointDataset(df_train.loc[:, features], label_train, WINDOW_SIZE, negative_to_positive_ration=100)
    ds_test = ChangePointDataset(df_test.loc[:, features], label_test, WINDOW_SIZE, negative_to_positive_ration=200)
    dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                  persistent_workers=True, pin_memory=True)
    dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                persistent_workers=True, pin_memory=True)
    dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS)  # bit of leftover

    print("Start fitting...")

    # get actual model
    model = LitChangePointClassifier(WINDOW_SIZE, len(features))
    early_stop_callback = EarlyStopping(monitor="val_f2", mode="max", patience=5)
    trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                        callbacks=[early_stop_callback], check_val_every_n_epoch=10, accumulate_grad_batches=5)
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model=model, dataloaders=dataloader_test)

    # store model
    trainer.save_checkpoint("../trained_model/classification.ckpt")

    return


if __name__ == "__main__":
    main()
