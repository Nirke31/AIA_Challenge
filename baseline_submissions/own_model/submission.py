import gc
import logging
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

from typing import List

from myModel import Seq2SeqTransformer, create_mask
from dataset_manip import MyDataset, load_data, convert_tgts_for_eval, split_train_test, pad_sequence_vec
from baseline_submissions.evaluation import NodeDetectionEvaluator

# torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
# https://github.com/pytorch/pytorch/issues/110213

def collate_fn(batch):
    src_batch, tgt_batch, objectID = [], [], []
    for src, tgt, id in batch:
        src_batch.append(src)
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
    train_df, test_df = split_train_test(data, train_test_ration=TRAIN_TEST_RATION, random_state=RANDOM_STATE)
    test_df = train_df.copy()  # FOR DEBUGGING ONLY
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
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, collate_fn=collate_fn)
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
        # print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json")

        loss.append(train_loss)
    print("Training done")

    if store_model:
        torch.save(transformer.state_dict(), TRAINED_MODEL_PATH)
        print("Model stored")

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
        _, asdf = torch.max(pred, dim=-1)

        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_expected.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(dl)


@torch.no_grad()
def greedy_decode(ds: MyDataset, model: Seq2SeqTransformer):
    model.eval()

    tgt_dict = ds.tgt_dict_int_to_str
    dl_eval = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    # store pred and tgt in list to concatenate later. This should make things faster at the expense of RAM?
    pred_df_all = []
    tgt_df_all = []

    i = 1
    for src, tgt, objectIds in dl_eval:
        print(f'\rEvaluation: {i}/{len(dl_eval)}', end='')
        tgt = tgt[:, 1:-1, :]  # remove BOS and EOS tokens

        src = src.to(DEVICE)
        src_seq_len = src.size(dim=1)
        src_mask = torch.zeros((1 * NHEAD, src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        tgt_seq_len = src_seq_len + 1  # BOS + seq len
        temp = torch.zeros((1, tgt_seq_len, 1), dtype=tgt.dtype, device=DEVICE)
        tgt_input = temp
        # tgt_input = torch.tensor([[[BOS]]], dtype=torch.long, device=DEVICE)
        tgt_input[:, 0, :] = BOS

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=DEVICE)
        tgt_mask = tgt_mask.unsqueeze(0).repeat(1 * NHEAD, 1, 1)

        memory = model.encode(src, src_mask)
        memory.to(DEVICE)

        # iterate over the whole sequence length and predict next tgt token
        for cur_seq_len in range(1, src_seq_len + 1):
            out = model.decode(tgt_input[:, :cur_seq_len, :], memory, tgt_mask[:, :cur_seq_len, :cur_seq_len])
            pred = model.generator(out[:, -1])
            # convert class to number, get only the last predicted class
            _, next_token = torch.max(pred, dim=1)
            next_token = next_token.item()

            tgt_input[:, cur_seq_len, :] = next_token
            # tgt_input = torch.cat([tgt_input, torch.ones(1, 1, 1).type_as(tgt_input.data).fill_(next_token)], dim=1)

        # if we iterated through the whole sequence we have the predicted tgt sequence in tgt input
        # remove BOS token
        tgt_input = tgt_input[:, 1:, :]
        pred_df, tgt_df = convert_tgts_for_eval(tgt_input, tgt, objectIds, tgt_dict)
        pred_df_all.append(pred_df)
        tgt_df_all.append(tgt_df)
        i += 1

    pred_df = pd.concat(pred_df_all)
    tgt_df = pd.concat(tgt_df_all)
    print('\n')

    pred_df.to_csv(Path("evaluations/pred.csv"), index=False)
    tgt_df.to_csv(Path("evaluations/tgt.csv"), index=False)
    print("Evaluation stored")

    evaluator = NodeDetectionEvaluator(tgt_df, pred_df, 6)
    return evaluator


# Learning settings
NUM_CSV_SETS = 2  # -1 = all
TRAIN_TEST_RATION = 0.8
BATCH_SIZE = 1
NUM_EPOCHS = 50
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
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NHEAD = 8
SRC_SIZE = 16  # this size has to be divisble by NHEADS or something like that?
TGT_SIZE = 35
EMB_SIZE = 512
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
LOAD_EVAL = False
RANDOM_STATE = 42
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

torch.set_printoptions(profile="full")

if __name__ == "__main__":
    # get everything
    ds_train, ds_test, transformer, loss_fn, optimizer = load_model_and_datasets(NUM_CSV_SETS)

    # train or load
    if not LOAD_EVAL:
        if LOAD_MODEL:
            transformer.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        else:
            losses = train(ds_train, transformer, optimizer)
            # plt.plot(losses)
            # plt.show(block=False)
            # time.sleep(1)

    # for param in transformer.parameters():
    #     print(param.max(), param.min())

    print("Evaluation:")
    if LOAD_EVAL:
        tgt = pd.read_csv(Path("evaluations/tgt.csv"))
        pred = pd.read_csv(Path("evaluations/pred.csv"))
        evaluatinator = NodeDetectionEvaluator(tgt, pred, 6)
    else:
        evaluatinator = greedy_decode(ds_test, model=transformer)

    precision, recall, f2, rmse = evaluatinator.score(debug=False)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {rmse:.2f}')

    evaluatinator.plot(object_id=1335)
