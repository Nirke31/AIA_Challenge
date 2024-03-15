from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from own_model.src.myModel import Autoencoder


def get_encoded_features(df: pd.DataFrame) -> torch.Tensor:
    model: Autoencoder = Autoencoder.load_from_checkpoint("../trained_model/autoencoder.ckpt")
    device = model.device
    data = torch.from_numpy(df.to_numpy(dtype=np.float32))
    out = model.encode_features(data.to(device)).cpu()
    return out


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


FEATURES_EW = [
    # "Eccentricity",
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

ENGINEERED_FEATURES_EW = {
    ("std", lambda x: x.rolling(window=6).std()):
        ["Semimajor Axis (m)", "Altitude (m)"],  # , "RAAN (deg)",, "Eccentricity"
}
ENGINEERED_FEATURES_NS = {
    ("std", lambda x: x.rolling(window=6).std()):
        ["Semimajor Axis (m)", "Altitude (m)"],
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

TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train_labels.csv")

RANDOM_STATE = 42
SHUFFLE = True
TRAIN_TEST_RATIO = 0.8
BATCH_SIZE = 512 * 4
EPOCHS = 300
DIRECTION = "NS"
NUM_WORKERS = 8
NUM_CSV_SETS = -1
FEATURES = FEATURES_EW if DIRECTION == "EW" else FEATURES_NS

if __name__ == "__main__":
    L.seed_everything(RANDOM_STATE, workers=True)
    # df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # df.to_pickle("../../dataset/df.pkl")
    df: pd.DataFrame = pd.read_pickle("../../dataset/df.pkl")

    # unwrap
    df[DEG_FEATURES] = np.unwrap(np.deg2rad(df[DEG_FEATURES]))

    # FEATURE ENGINEERING
    feature_dict = ENGINEERED_FEATURES_EW if DIRECTION == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_" + DIRECTION
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF, backfill to fill NANs resulting from window
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()
            FEATURES.append(new_feature_name)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # add lags
    df, FEATURES = add_lag_features(df, FEATURES, 4)
    # test = get_encoded_features(df[FEATURES])

    data = torch.from_numpy(df[FEATURES].to_numpy(dtype=np.float32))
    data_train, data_test = train_test_split(data, test_size=1 - TRAIN_TEST_RATIO,
                                             random_state=RANDOM_STATE, shuffle=SHUFFLE)
    ds_train = TensorDataset(data_train)
    ds_test = TensorDataset(data_test)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS,
                          persistent_workers=True, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                         persistent_workers=True, pin_memory=True)

    SRC_SIZE = len(FEATURES)
    autoencoder = Autoencoder(SRC_SIZE, 512, 20, act_fn=nn.Tanh)
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          mode="min",
                                          dirpath="lightning_logs/best_checkpoints",
                                          filename='autoencoder_{epoch:02d}_{val_loss:.2f}')
    trainer = Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                      callbacks=[early_stop_callback, checkpoint_callback],
                      check_val_every_n_epoch=1)  # , accumulate_grad_batches=3

    # Train the model
    trainer.fit(autoencoder, dl_train, dl_test)

    trainer.save_checkpoint("../trained_model/autoencoder.ckpt")
