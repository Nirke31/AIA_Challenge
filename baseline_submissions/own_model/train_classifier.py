import math
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import lightning as L

from baseline_submissions.own_model.dataset_manip import load_data_window_ready, GetWindowDataset
from baseline_submissions.own_model.myModel import LitClassifier

SHUFFLE_DATA = False
FEATURES = ["Eccentricity",
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
            # "Vz (m/s)"
            ]

# User settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = False
LOAD_EVAL = False
RANDOM_STATE = 42
TRAINED_MODEL_NAME = "model.pkl"
TRAINED_MODEL_PATH = Path('./trained_model/' + TRAINED_MODEL_NAME)
TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")

NUM_CSV_SETS = -1
SRC_SIZE = len(FEATURES)
TGT_SIZE = 5  # based on the dataset dict
TRAIN_TEST_RATIO = 0.8
TRAIN_VAL_RATION = 0.8
BATCH_SIZE = 5
WINDOW_SIZE = 11
EPOCHS = 200

if __name__ == "__main__":
    # FOR FITTING WINDOW MODEL
    data, labels = load_data_window_ready(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, NUM_CSV_SETS)
    data = data[FEATURES]
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    EW_labels = labels[labels["Direction"] == "EW"]
    NS_labels = labels[labels["Direction"] == "NS"]
    # shuffle targets
    labels = labels.sample(frac=1)
    labels.reset_index(drop=True, inplace=True)

    split_train_test = math.floor(labels.shape[0] * TRAIN_TEST_RATIO)
    split_train_val = math.floor(split_train_test * TRAIN_VAL_RATION)

    train_labels = labels.iloc[:split_train_val]
    val_labels = labels.iloc[split_train_val:split_train_test]
    test_labels = labels.iloc[split_train_test:]

    test_labels.reset_index(drop=True, inplace=True)
    val_labels.reset_index(drop=True, inplace=True)
    # HAVE TO IMPROVE THIS ABOVE OR PUT INTO FUNCTION

    # test_df = train_df.copy()  # FOR DEBUGGING ONLY
    ds_train = GetWindowDataset(data, train_labels, WINDOW_SIZE)
    ds_val = GetWindowDataset(data, val_labels, WINDOW_SIZE)
    ds_test = GetWindowDataset(data, test_labels, WINDOW_SIZE)
    dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
    dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
    dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

    print("Start fitting...")

    # get actual model
    model = LitClassifier(WINDOW_SIZE, SRC_SIZE + 1, TGT_SIZE)
    early_stop_callback = EarlyStopping(monitor="val_f2", mode="max", patience=5)
    trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True,
                        callbacks=[early_stop_callback], check_val_every_n_epoch=10)
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(model=model, dataloaders=dataloader_test)



