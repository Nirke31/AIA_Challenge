import pandas as pd
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from myModel import Seq2SeqTransformer, create_mask
from dataset_manip import MyDataset, load_data

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

# TODO: VALUES TO BE DONE
SRC_PADDING_VEC = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
TGT_PADDING_VEC = torch.Tensor([14, 15])


def train_epoch(dl: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ):
    model.train()
    model.to(DEVICE)
    losses = 0

    for src, tgt in dl:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
         memory_mask, memory_padding_mask) = create_mask(src, tgt, NHEAD, SRC_PADDING_VEC, TGT_PADDING_VEC, DEVICE)

        pred = model(src, tgt, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, memory_padding_mask)

        optimizer.zero_grad()

        loss = loss_fn(pred.reshape(-1, pred.shape[-1], tgt.shape[-1]))
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
NHEAD = 5
SRC_SIZE = len(FEATURE_COLS)
TGT_SIZE = 7  # TB updated
FF_SIZE = 512
DROPOUT = 0.1
LR = 0.0001
BETAS = (0.9, 0.98)
EPS = 1e-9
WEIGHT_DECAY = 0  # For now keep as ADAM, default

if __name__ == "__main__":

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NHEAD, SRC_SIZE, TGT_SIZE, FF_SIZE,
                                     DROPOUT)

    loss_fn = torch.nn.CrossEntropyLoss()  # ignore_index=TBD ?

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

    train_data_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train"
    train_label_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv"
    data: pd.DataFrame = load_data(train_data_str, train_label_str)

    ds = MyDataset(data)
    dl = DataLoader(ds)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(dl, transformer, optimizer)
        end_time = timer()
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
