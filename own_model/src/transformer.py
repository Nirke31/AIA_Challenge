import gc
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, fbeta_score, precision_score, make_scorer

from typing import List, Tuple

from baseline_submissions.ml_python import utils
from myModel import TransformerINATOR
from dataset_manip import MyDataset, load_data, convert_tgts_for_eval, split_train_test, pad_sequence_vec, \
    state_change_eval, load_data_window_ready, GetWindowDataset
# sys.path.append(os.path.abspath("baseline_submissions"))  # so that vscode findest the NodeDetectionEvaluator
from baseline_submissions.evaluation import NodeDetectionEvaluator

# torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# torch.backends.cudnn.benchmark = True

# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
# https://pytoch.org/docs/stable/generated/torch.nn.Transformer.html
# https://github.com/pytorch/pytorch/issues/110213
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# same problem I have?


def collate_fn(batch):
    # probably a smarter way but whatever
    src_batch, tgt_batch, objectID_batch, timeIndex_batch = [], [], [], []
    for src, tgt, id, timeIndex in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
        objectID_batch.append(id)
        timeIndex_batch.append(timeIndex)

    src_batch = pad_sequence_vec(src_batch, padding_vec=SRC_PADDING_VEC)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=TGT_PADDING_NBR)

    return src_batch, tgt_batch, torch.tensor(objectID_batch), torch.tensor(timeIndex_batch)


def load_datasets(train_test_ratio: float, random_state: int, amount: int = 10):
    print("Loading Dataframe")
    data: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount)
    data = data[FEATURES_AND_TGT]
    print("Dataframe load done")

    # get Dataset
    train_df, test_df = split_train_test(data, train_test_ration=train_test_ratio, random_state=random_state)
    # test_df = train_df.copy()  # FOR DEBUGGING ONLY
    ds_train = MyDataset(train_df)
    ds_test = MyDataset(test_df)
    print("Own Dataset created\n")

    return ds_train, ds_test


# Learning settings
NUM_CSV_SETS = 50  # -1 = all
TRAIN_TEST_RATIO = 0.8
BATCH_SIZE = 5
NUM_EPOCHS = 150
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
# Transformer settings
NHEAD = 8
SRC_SIZE = len(FEATURES)
TGT_SIZE = 5  # THIS IS BASED ON THE DATASET DICT PLUS ONE PADDING !!!!
EMB_SIZE = 64  # this size has to be divisble by NHEADS or something like that?
DIM_HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.1
# Optimizer settings
LR = 0.001
BETAS = (0.9, 0.98)
EPS = 1e-9
WEIGHT_DECAY = 0  # For now keep as ADAM, default
# User settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = False
LOAD_EVAL = False
RANDOM_STATE = 42
TRAINED_MODEL_NAME = "model.pkl"
TRAINED_MODEL_PATH = Path('./trained_model/' + TRAINED_MODEL_NAME)
TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
# Padding
TGT_PADDING_NBR = 4  # got from MyDataset dict
SRC_PADDING_VEC = torch.zeros(SRC_SIZE)

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
    split_ier = math.floor(labels.shape[0] * TRAIN_TEST_RATIO)
    train_labels = labels.iloc[:split_ier]
    test_labels = labels.iloc[split_ier:]
    test_labels.reset_index(drop=True, inplace=True)
    # HAVE TO IMPROVE THIS ABOVE OR PUT INTO FUNCTION

    # test_df = train_df.copy()  # FOR DEBUGGING ONLY
    window_size = 11
    ds_train = GetWindowDataset(data, train_labels, window_size)
    ds_test = GetWindowDataset(data, test_labels, window_size)
    dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, collate_fn=collate_fn)
    dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, collate_fn=collate_fn)

    model = TransformerINATOR(SRC_SIZE, EMB_SIZE, TGT_SIZE, NHEAD, window_size,
                              DIM_HIDDEN, N_LAYERS, DROPOUT, True)
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TGT_PADDING_NBR)
    optimizer = torch.optim.RAdam(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(model.parameters())

    # train or load
    if not LOAD_EVAL:
        if LOAD_MODEL:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        else:
            losses = model.do_train(dataloader_train, NUM_EPOCHS, optimizer, loss_fn, SRC_PADDING_VEC,
                                    DEVICE, True, TRAINED_MODEL_PATH)
            plt.plot(losses)
            plt.show(block=True)
            # time.sleep(1)

    print("Evaluation:")
    if LOAD_EVAL:
        tgt = pd.read_csv(Path("evaluations/tgt.csv"))
        pred = pd.read_csv(Path("evaluations/pred.csv"))
        evaluatinator = NodeDetectionEvaluator(tgt, pred, 6)
    else:
        loss = model.do_test(dataloader_test, loss_fn, SRC_PADDING_VEC, DEVICE)
        print(f"Loss over all test sequences: {loss}")

    # precision, recall, f2, rmse = evaluatinator.score(debug=True)
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F2: {f2:.2f}')
    # print(f'RMSE: {rmse:.2f}')
    #
    # evaluatinator.plot(object_id=1335)
