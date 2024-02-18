import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from myModel import Seq2SeqTransformer, create_mask
from dataset_manip import MyDataset, load_data

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)  # 'False': fixes cuda problem but increases runtime by a gazillion
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=False)
# "enable_math": False, "enable_flash": True, "enable_mem_efficient": False

# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

# TODO: VALUES TO BE DONE, should rely on actul src and tgt size
SRC_PADDING_VEC = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
TGT_PADDING_VEC = torch.Tensor([15])

# https://github.com/pytorch/pytorch/issues/110213


def train_epoch(dl: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ):
    model.train()
    losses = 0

    for src, tgt in dl:
        # mask is tuple, all masks are pushed to DEVICE
        masks = create_mask(src, tgt, NHEAD, SRC_PADDING_VEC, TGT_PADDING_VEC, DEVICE)

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        pred = model(src, tgt, *masks)

        optimizer.zero_grad()

        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(dl)


def evaluate(transformer):
    return


FEATURE_COLS = [
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
    "Vz (m/s)"
]

# DEFINES
NUM_EPOCHS = 5
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NHEAD = 8
SRC_SIZE = 16
TGT_SIZE = -1  # updated in code down below
FF_SIZE = 512
DROPOUT = 0.1
LR = 0.0001
BETAS = (0.9, 0.98)
EPS = 1e-9
WEIGHT_DECAY = 0  # For now keep as ADAM, default

if __name__ == "__main__":
    train_data_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train"
    train_label_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv"
    data: pd.DataFrame = load_data(train_data_str, train_label_str)

    ds = MyDataset(data)
    tgt_dict = ds.tgt_dict
    TGT_SIZE = len(tgt_dict)  # number of different targets

    dl = DataLoader(ds)

    # Create transformer
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NHEAD, SRC_SIZE,
                                     TGT_SIZE, FF_SIZE, DROPOUT)
    # idk if I really need this. shouldn't torch do this already?
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()  # ignore_index=TBD ?

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(dl, transformer, optimizer)
        end_time = timer()
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
