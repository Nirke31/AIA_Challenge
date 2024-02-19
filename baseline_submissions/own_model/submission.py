import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from timeit import default_timer as timer
from myModel import Seq2SeqTransformer, create_mask
from dataset_manip import MyDataset, load_data, convert_tgts_for_eval
from baseline_submissions.evaluation import NodeDetectionEvaluator

# torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
# https://github.com/pytorch/pytorch/issues/110213

# TODO: VALUES TO BE DONE, should rely on actul src and tgt size
SRC_PADDING_VEC = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
TGT_PADDING_VEC = torch.Tensor([15])


def train_epoch(dl: DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ):
    model.train()
    losses = 0

    for src, tgt, objectIds in dl:
        # mask is tuple, all masks are already pushed to DEVICE
        masks = create_mask(src, tgt, NHEAD, SRC_PADDING_VEC, TGT_PADDING_VEC, DEVICE)

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_interference"):
        pred = model(src, tgt, *masks)

        optimizer.zero_grad()

        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return losses / len(dl)


def simple_eval(ds: MyDataset, model: torch.nn.Module):
    model.eval()

    tgt_dict = ds.tgt_dict
    dl_eval = DataLoader(ds, batch_size=1)

    pred_all = []
    tgt_all = []
    objectIDs_all = []
    for src, tgt, objectId in dl_eval:
        # mask is tuple, all masks are already pushed to DEVICE
        masks = create_mask(src, tgt, NHEAD, SRC_PADDING_VEC, TGT_PADDING_VEC, DEVICE)

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        pred = model(src, tgt, *masks)
        pred_all.append(pred)
        tgt_all.append(tgt)
        objectIDs_all.append(objectId)

    whole_pred = torch.cat(pred_all, dim=0)
    whole_tgt = torch.cat(tgt_all, dim=0)
    whole_ids = torch.cat(objectIDs_all, dim=0)

    pred_df, tgt_df = convert_tgts_for_eval(whole_pred, whole_tgt, whole_ids, tgt_dict)
    # pred_df.to_csv("pred.csv")
    # tgt_df.to_csv("tgt.csv")

    evaluator = NodeDetectionEvaluator(tgt_df, pred_df, 6)
    precision, recall, f2, rmse = evaluator.score(debug=False)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {rmse:.2f}')
    return


# DEFINES
NUM_EPOCHS = 10
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
    print("Loading Dataframe")
    data: pd.DataFrame = load_data(train_data_str, train_label_str)
    print("Dataframe load done")

    # get Dataset
    ds = MyDataset(data)
    TGT_SIZE = len(ds.tgt_dict)  # number of different targets
    print("Own Dataset created")

    # create Dataloader
    dl = DataLoader(ds, batch_size=1)
    print("Dataloader created")

    # Create transformer
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NHEAD, SRC_SIZE,
                                     TGT_SIZE, FF_SIZE, DROPOUT)
    # idk if I really need this. shouldn't torch do this already?
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    transformer.to(DEVICE)
    print("Transformer created")

    loss_fn = torch.nn.CrossEntropyLoss()  # ignore_index=TBD ?

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

    # Train
    print("Starting Training: ----------------------------------------------------")
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(dl, transformer, optimizer)
        end_time = timer()
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

    print("Training done")

    print("Storing model")
    torch.save(transformer.state_dict(), Path('./trained_model/model.pkl'))

    print("Evaluation...")
    simple_eval(ds, model=transformer)
