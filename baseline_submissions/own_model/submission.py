import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from typing import List, Tuple

from myModel import TransformerINATOR
from dataset_manip import MyDataset, load_data, convert_tgts_for_eval, split_train_test, pad_sequence_vec
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
    src_batch, tgt_batch, objectID = [], [], []
    for src, tgt, id in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
        objectID.append(id)

    src_batch = pad_sequence_vec(src_batch, padding_vec=SRC_PADDING_VEC)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=TGT_PADDING_NBR)

    return src_batch, tgt_batch, torch.tensor(objectID)


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
NUM_CSV_SETS = 5  # -1 = all
TRAIN_TEST_RATIO = 0.8
BATCH_SIZE = 1
NUM_EPOCHS = 200
SHUFFLE_DATA = False
FEATURES_AND_TGT = [
    "Timestamp",
    "Eccentricity",
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
    "Vz (m/s)",
    "EW/NS"
]
# Transformer settings
NHEAD = 16
SRC_SIZE = len(FEATURES_AND_TGT) - 1  # Features minus the target (16 for all features)
TGT_SIZE = 33  # THIS IS BASED ON THE DATASET DICT PLUS ONE PADDING !!!!
EMB_SIZE = 128  # this size has to be divisble by NHEADS or something like that?
DIM_HIDDEN = 2048
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
TGT_PADDING_NBR = 32  # got from MyDataset dict
SRC_PADDING_VEC = torch.zeros(SRC_SIZE)


if __name__ == "__main__":
    # create everything
    dataset_train, dataset_test = load_datasets(TRAIN_TEST_RATIO, RANDOM_STATE, NUM_CSV_SETS)
    dataset_test = dataset_test
    dataloader_train = DataLoader(dataset_train, BATCH_SIZE, SHUFFLE_DATA, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, BATCH_SIZE, SHUFFLE_DATA, collate_fn=collate_fn)

    model = TransformerINATOR(SRC_SIZE, EMB_SIZE, TGT_SIZE, NHEAD, DIM_HIDDEN, N_LAYERS, DROPOUT, True)
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TGT_PADDING_NBR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

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
        evaluatinator, loss = model.do_test(dataloader_test, loss_fn, SRC_PADDING_VEC, DEVICE)
        print(f"Loss over all test sequences: {loss}")

    precision, recall, f2, rmse = evaluatinator.score(debug=False)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {rmse:.2f}')

    evaluatinator.plot(object_id=1335)
