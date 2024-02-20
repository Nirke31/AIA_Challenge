import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from typing import List

from myModel import Seq2SeqTransformer, create_mask
from dataset_manip import MyDataset, load_data, convert_tgts_for_eval, split_train_test, pad_sequence_vec
from baseline_submissions.evaluation import NodeDetectionEvaluator

# torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
# https://github.com/pytorch/pytorch/issues/110213

def collate_fn(batch):
    src_batch, tgt_batch, objectID = [], [], []
    for src, tgt, id in batch:
        src_batch.append(torch.cat((src, SRC_EOS_VEC)))
        tgt_batch.append(torch.cat((torch.tensor([[BOS]]), tgt, torch.tensor([[EOS]]))))
        objectID.append(id)

    src_batch = pad_sequence_vec(src_batch, padding_vec=SRC_PADDING_VEC)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=TGT_PADDING_NUMBER)

    return src_batch, tgt_batch, torch.tensor(objectID)


def load_model_and_datasets(amount: int = 10):
    print("Loading Dataframe")
    data: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount)
    data = data[FEATURES_AND_TGT]
    print("Dataframe load done")

    # get Dataset
    train_df, test_df = split_train_test(data, train_test_ration=TRAIN_TEST_RATION)
    ds_train = MyDataset(train_df)
    ds_test = MyDataset(test_df)
    print("Own Dataset created")

    # Create transformer
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NHEAD, SRC_SIZE,
                                     TGT_SIZE, FF_SIZE, DROPOUT)
    # idk if I really need this. shouldn't torch do this already?
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    transformer.to(DEVICE)
    print("Transformer created")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TGT_PADDING_NUMBER)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

    return ds_train, ds_test, transformer, loss_fn, optimizer


def train(ds: MyDataset, transformer: torch.nn.Module, optimizer: torch.optim.Optimizer, store_model: bool = True) -> \
        List[float]:
    # create Dataloader
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    print("Dataloader created")

    loss = []
    print("Starting Training: ----------------------------------------------------")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #     with record_function("model_interference"):
        #         train_loss = train_epoch(dl, transformer, optimizer)
        train_loss = train_epoch(dl, transformer, optimizer)
        end_time = timer()
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json")

        loss.append(train_loss)
    print("Training done")

    if store_model:
        print("Storing model")
        torch.save(transformer.state_dict(), TRAINED_MODEL_PATH)

    return loss


def train_epoch(dl: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    losses = 0

    for src, tgt, objectIds in dl:
        # Shift tgt such that the model starts with the BOS token and predicts based on that token.
        tgt_input = tgt[:, :-1, :]  # batch len, sequence length, feature length
        tgt_expected = tgt[:, 1:, :]
        # mask is tuple, all masks are already pushed to DEVICE
        masks = create_mask(src, tgt_input, NHEAD, SRC_PADDING_VEC, TGT_PADDING_VEC, DEVICE,
                            [False, True, False, True, True, False])

        src = src.to(DEVICE)
        tgt_input = tgt_input.to(DEVICE)
        tgt_expected = tgt_expected.to(DEVICE)

        pred = model(src, tgt_input, *masks)

        optimizer.zero_grad()

        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_expected.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(dl)


def simple_eval(ds: MyDataset, model: torch.nn.Module):
    model.eval()

    tgt_dict = ds.tgt_dict_int_to_str
    dl_eval = DataLoader(ds, batch_size=1)

    # store pred and tgt in list to concatenate later. This should make things faster at the expense of RAM?
    pred_df_all = []
    tgt_df_all = []
    for src, tgt, objectIds in dl_eval:
        # mask is tuple, all masks are already pushed to DEVICE
        masks = create_mask(src, tgt, NHEAD, SRC_PADDING_VEC, TGT_PADDING_VEC, DEVICE,
                            [False, True, False, True, True, False])

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        pred = model(src, tgt, *masks)
        pred_df, tgt_df = convert_tgts_for_eval(pred, tgt, objectIds, tgt_dict)
        pred_df_all.append(pred_df)
        tgt_df_all.append(tgt_df)

    pred_df = pd.concat(pred_df_all)
    tgt_df = pd.concat(tgt_df_all)
    # pred_df.to_csv("pred.csv")
    # tgt_df.to_csv("tgt.csv")

    evaluator = NodeDetectionEvaluator(tgt_df, pred_df, 6)
    return evaluator


# Learning settings
NUM_CSV_SETS = 15  # -1 = all
TRAIN_TEST_RATION = 0.75
BATCH_SIZE = 1
NUM_EPOCHS = 5
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
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NHEAD = 8
SRC_SIZE = 16  # this size has to be divisble by NHEADS or something like that?
TGT_SIZE = 35
FF_SIZE = 512
DROPOUT = 0.1
# Optimizer settings
LR = 0.0001
BETAS = (0.9, 0.98)
EPS = 1e-9
WEIGHT_DECAY = 0  # For now keep as ADAM, default
# User settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = False
TRAINED_MODEL_NAME = "model.pkl"
TRAINED_MODEL_PATH = Path('trained_model/' + TRAINED_MODEL_NAME)
TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
# Things
BOS = 32  # Beginning of Sequence
EOS = 33  # End of Sequence
SRC_PADDING_VEC = torch.zeros(SRC_SIZE)
SRC_EOS_VEC = torch.ones(1, SRC_SIZE)  # I am unsure if I need this
TGT_PADDING_NUMBER = 34
TGT_PADDING_VEC = torch.Tensor([TGT_PADDING_NUMBER])

if __name__ == "__main__":

    # get everything
    ds_train, ds_test, transformer, loss_fn, optimizer = load_model_and_datasets(NUM_CSV_SETS)

    # train or load
    if LOAD_MODEL:
        transformer.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    else:
        losses = train(ds_train, transformer, optimizer)
        # plt.plot(losses)
        # plt.show()

    # print("Evaluation...")
    # evaluatinator = simple_eval(ds_test, model=transformer)
    #
    # precision, recall, f2, rmse = evaluatinator.score(debug=False)
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F2: {f2:.2f}')
    # print(f'RMSE: {rmse:.2f}')

    # evaluatinator.plot(object_id=1157)
