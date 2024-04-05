import math
from pathlib import Path

import numpy as np
from joblib import dump

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L

from own_model.src.dataset_manip import load_data_window_ready, GetWindowDataset
from own_model.src.myModel import LitClassifier

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
    "Altitude (m)",
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
    "Altitude (m)",
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
         "Altitude (m)", ],
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
         "Altitude (m)", ],
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
TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train_labels.csv")

TGT_SIZE = 5  # based on the dataset dict
TRAIN_TEST_RATIO = 0.90
BATCH_SIZE = 20
WINDOW_SIZE = 101
EPOCHS = 400
DIRECTION = "NS"
FIRST = True
NUM_WORKERS = 2
NUM_CSV_SETS = -1
FEATURES = FEATURES_EW if DIRECTION == "EW" else FEATURES_NS
SRC_SIZE = len(FEATURES)

if __name__ == "__main__":
    L.seed_everything(RANDOM_STATE, workers=True)
    # FOR FITTING WINDOW MODEL
    data, labels = load_data_window_ready(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, NUM_CSV_SETS)
    # data.to_pickle("../../dataset/data.pkl")
    # labels.to_pickle("../../dataset/labels.pkl")
    # data: pd.DataFrame = pd.read_pickle("../../dataset/data.pkl")
    # labels: pd.DataFrame = pd.read_pickle("../../dataset/labels.pkl")

    # Train only first sample or without first sample
    if FIRST:
        labels = labels[labels['TimeIndex'] == 0]
    else:
        labels = labels[labels['TimeIndex'] != 0]

    # unwrap
    data[DEG_FEATURES] = np.unwrap(np.deg2rad(data[DEG_FEATURES]))

    # FEATURE ENGINEERING
    feature_dict = ENGINEERED_FEATURES_EW if DIRECTION == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF, backfill to fill NANs resulting from window
            data[new_feature_name] = data.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()
            FEATURES.append(new_feature_name)

    data = data[FEATURES]

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    dump(scaler, "../trained_model/scaler.joblib", compress=0)

    EW_labels = labels[labels["Direction"] == "EW"]
    NS_labels = labels[labels["Direction"] == "NS"]
    # shuffle targets
    labels = EW_labels.copy() if DIRECTION == "EW" else NS_labels.copy()  # sets here which direction we are training
    labels = labels.sample(frac=1)
    labels.reset_index(drop=True, inplace=True)

    split_train_test = math.floor(labels.shape[0] * TRAIN_TEST_RATIO)

    train_labels = labels.iloc[:split_train_test]
    val_labels = labels.iloc[split_train_test:]
    val_labels.reset_index(drop=True, inplace=True)
    # HAVE TO IMPROVE THIS ABOVE OR PUT INTO FUNCTION

    ds_train = GetWindowDataset(data, train_labels, WINDOW_SIZE)
    ds_val = GetWindowDataset(data, val_labels, WINDOW_SIZE)
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
    dataloader_test = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=NUM_WORKERS)

    print("Start fitting...")

    # get actual model
    # have to (re)set size because features were added
    SRC_SIZE = len(FEATURES)
    model = LitClassifier(WINDOW_SIZE, SRC_SIZE, 1e-4, TGT_SIZE)
    early_stop_callback = EarlyStopping(monitor="val_MulticlassFBetaScore", mode="max", patience=40)
    checkpoint_callback = ModelCheckpoint(monitor="val_MulticlassFBetaScore",
                                          mode="max",
                                          dirpath="lightning_logs/best_checkpoints",
                                          filename='classification_{epoch:02d}_{val_MulticlassFBetaScore:.2f}')
    trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        check_val_every_n_epoch=1)  # , accumulate_grad_batches=3
    # LR finder, didnt really help me for first sample
    # tuner = Tuner(trainer)
    # # Run learning rate finder
    # lr_finder = tuner.lr_find(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    # # Results can be found in
    # print(lr_finder.results)
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # # update hparams of the model
    # model.hparams.lr = new_lr

    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model=model, dataloaders=dataloader_test)

    # store model
    trainer.save_checkpoint("../trained_model/classification.ckpt")
