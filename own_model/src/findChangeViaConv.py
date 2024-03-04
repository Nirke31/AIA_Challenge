import math
import pickle
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
from torch.utils.data import DataLoader, WeightedRandomSampler

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
NUM_WORKERS = 2
EPOCHS = 200
BATCH_SIZE = 1
SHUFFLE_DATA = False
WINDOW_SIZE = 25
DIRECTION = "EW"
NUM_CSV_SETS = -1

L.seed_everything(RANDOM_STATE, workers=True)


# https://github.com/geekfeiw/Multi-Scale-1D-ResNet/blob/master/figs/network.png
# https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

def main():
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0.0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0.0

    # features, leaves room for potential feature engineering
    features = BASE_FEATURES_EW if DIRECTION == "EW" else BASE_FEATURES_NS
    # Scale, yes I am leaking, would be way more complicated to not leak. Maybe change later anyways?
    scaler = StandardScaler()
    df.loc[:, features] = pd.DataFrame(scaler.fit_transform(df.loc[:, features]), index=df.index, columns=features)

    print("Dataset loaded")

    # split train test, DATA IS SHUFFLED!
    labels = df.loc[:, DIRECTION].to_numpy()
    indices = np.arange(labels.shape[0])
    idx_train, idx_val = train_test_split(indices, train_size=TRAIN_TEST_RATIO, stratify=labels)

    # load dataset and dataloader
    ds_train = ChangePointDataset(df.loc[:, features], labels, idx_train, WINDOW_SIZE)
    ds_test = ChangePointDataset(df.loc[:, features], labels, idx_val, WINDOW_SIZE)

    # Create random sampler to sample more positives than negatives
    num_samples = ds_train.tgt.shape[0]
    num_pos = ds_train.tgt.sum()
    num_neg = num_samples - ds_train.tgt.sum()
    class_counts = {1: num_pos, 0: num_neg}
    # each sample is assigned a weight based on its class
    weights = [1 / class_counts[i] for i in ds_train.tgt]
    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)

    dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                  persistent_workers=True, pin_memory=False, sampler=sampler)
    # For val (and test) we do not need a sampler because we just look at all values once
    dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS,
                                persistent_workers=True, pin_memory=False)
    dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA,
                                 num_workers=NUM_WORKERS)  # bit of leftover

    print("Start fitting...")

    # get actual model
    early_stop_callback = EarlyStopping(monitor="val_BinaryFBetaScore", mode="max", patience=5)
    trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                        callbacks=[early_stop_callback], check_val_every_n_epoch=10, accumulate_grad_batches=5)
    model = LitChangePointClassifier(WINDOW_SIZE, len(features))
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model=model, dataloaders=dataloader_test)

    # store model
    trainer.save_checkpoint("../trained_model/classification.ckpt")

    return


if __name__ == "__main__":
    main()
