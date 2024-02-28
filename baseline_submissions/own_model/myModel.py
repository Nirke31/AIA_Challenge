from pathlib import Path
from typing import Optional, List, Callable, Any, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from timeit import default_timer as timer
import lightning as L
from torchmetrics.classification import Accuracy, FBetaScore, Recall, Precision

from baseline_submissions.evaluation import NodeDetectionEvaluator
from baseline_submissions.own_model.dataset_manip import convert_tgts_for_eval, MyDataset


def create_src_mask(src: Tensor, num_heads: int, device: torch.device) -> Tensor:
    batch_len: int = src.size(dim=0)
    seq_len: int = src.size(dim=1)
    src_mask = torch.zeros((batch_len * num_heads, seq_len, seq_len), device=device).type(torch.bool)
    return src_mask


def create_src_padding_mask(src: Tensor, src_pad_vector: Tensor, device: torch.device):
    src_num_features = src_pad_vector.shape[0]
    src_padding_mask = (src == src_pad_vector.view(1, 1, src_num_features)).all(dim=2).to(device=device)
    return src_padding_mask


#  helper Module copied from pytorch tutorial
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 3000):
        """
        Helper module for positional encoding. Copied from pytorch tutorial
        (https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model) and adapted to
        work with batch_size first
        Args:
            emb_size: int, embedding size
            dropout: float, dropout
            max_len:,int, max sequence length in the whole dataset
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        # batch_size = 1, len, emb
        pe = torch.zeros(1, max_len, emb_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerINATOR(nn.Module):
    def __init__(self, src_size: int, emb_size: int, tgt_size: int, nhead: int, window_size: int,
                 d_hid: int = 2048, nlayers: int = 2, dropout: float = 0.5, batch_first: bool = True):
        super().__init__()

        # layer stuff
        self.pos_encoder = PositionalEncoding(src_size, dropout, max_len=window_size)
        self.linear_in = nn.Linear(src_size, emb_size)

        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, d_hid, dropout, batch_first=batch_first,
                                                   activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.linear_out = nn.Linear(emb_size * window_size, tgt_size)

        self.init_weights()

        # variables
        self.mode_type = 'Transformer'
        self.src_size: int = src_size
        self.emb_size: int = emb_size
        self.tgt_size: int = tgt_size
        self.nhead: int = nhead
        self.d_hid: int = d_hid
        self.nlayers: int = nlayers

    def init_weights(self) -> None:
        # For now I do not initialse weights
        return
        # else
        # torch.nn.init.xavier_uniform_(self.linear_in.weight)
        # torch.nn.zeros_(self.linear_in.bias)

    def forward(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor) -> Tensor:
        """
        Args:
            src: input Tensor, shape ''[batch_size, seq_len, feature_size]''
            src_mask: Tensor, shape ''[seq_len, seq_len]''
            src_padding_mask: Tensor, shape ''[batch_size, seq_len]''

        Returns:
            output tensor of shape ''[batch_size, seq_len, tgt_size]''
        """
        src = self.linear_in(self.pos_encoder(src))
        # TUTORIAL HAS THIS. DO I REALLY NEED THIS???? FOR SOURCE IT SHOULDNT MATTER WHAT IT SEES RIGHT????
        # if src_mask is None:
        #     """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        #     Unmasked positions are filled with float(0.0).
        #     """
        #     src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.encoder(src, src_mask, src_padding_mask)
        output = self.linear_out(torch.flatten(output, start_dim=1))
        return output

    def do_train(self, dataloader: DataLoader, num_epochs: int, optimizer: torch.optim.Optimizer, loss_fn: Callable,
                 padding_vec: Tensor, device: torch.device, store_model: bool = True,
                 trained_model_path: Path = None) -> List[float]:

        loss = []
        print("Starting Training: ---------------------------------")
        for epoch in range(1, num_epochs + 1):
            start_time = timer()
            train_loss = self._train_epoch(dataloader, optimizer, loss_fn, padding_vec, device)
            end_time = timer()
            print(f"Epoch: {epoch}/{num_epochs}, "
                  f"Train loss: {train_loss:.3f}, "
                  f"Epoch time = {(end_time - start_time):.3f}s")
            loss.append(train_loss)

        print("Training done")

        if store_model:
            if Path is None:
                raise ValueError("store model is set true but not path was provided")
            torch.save(self.state_dict(), trained_model_path)
            print("Model stored")

        return loss

    def _train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: Callable,
                     padding_vec: Tensor, device: torch.device):
        self.train()
        losses = 0.0

        for src, tgt, objectIds, timeIndex in dataloader:
            src_mask = create_src_mask(src, self.nhead, device)
            src_padding_mask = create_src_padding_mask(src, padding_vec, device)

            src = src.to(device)
            tgt = tgt.to(device)

            pred = self.forward(src, src_mask, src_padding_mask)
            optimizer.zero_grad()

            test = pred.reshape(-1, pred.shape[-1])
            test1 = tgt.reshape(-1)
            loss = loss_fn(test, test1)
            loss.backward()

            # IDK if I need this?
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            losses += loss.item()

        # maybe this is dependents on the BATCH_SIZE
        return losses / len(dataloader)

    @torch.no_grad()
    def do_test(self, dataloader: DataLoader, loss_fn: Callable, padding_vec: Tensor, device: torch.device):
        self.eval()

        total_loss = 0.0
        pred_df_all = []
        tgt_df_all = []
        tgt_dict = dataloader.dataset.tgt_dict_int_to_str

        for src, tgt, objectIDs, timeIndex in dataloader:
            src_mask = create_src_mask(src, self.nhead, device)
            src_padding_mask = create_src_padding_mask(src, padding_vec, device)

            src = src.to(device)
            tgt = tgt.to(device)

            pred = self.forward(src, src_mask, src_padding_mask)
            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
            total_loss += loss.item()

            # TODO: into function
            tgt_np = tgt.numpy(force=True)  # 'most likely' a copy, forced of the GPU to cpu
            pred_np = pred.numpy(force=True)
            tgt_np = tgt_np.squeeze(-1)
            pred_np = np.argmax(pred_np, axis=-1)

            pred_df = pd.DataFrame(pred_np, columns=["Type"])
            tgt_df = pd.DataFrame(tgt_np, columns=["Type"])
            pred_df["ObjectID"] = objectIDs
            tgt_df["ObjectID"] = objectIDs
            pred_df["TimeIndex"] = timeIndex
            tgt_df["TimeIndex"] = timeIndex

            pred_df_all.append(pred_df)
            tgt_df_all.append(tgt_df)

        pred_df_all = pd.concat(pred_df_all)
        tgt_df_all = pd.concat(tgt_df_all)
        pred_df_all.loc[:, "Type"] = pred_df_all.loc[:, "Type"].map(dataloader.dataset.tgt_dict_int_to_str)
        tgt_df_all.loc[:, "Type"] = tgt_df_all.loc[:, "Type"].map(dataloader.dataset.tgt_dict_int_to_str)

        pred_df_all.to_csv(Path("./evaluations/pred.csv"), index=False)
        tgt_df_all.to_csv(Path("./evaluations/tgt.csv"), index=False)
        print("Evaluation stored")
        test = pred_df_all.loc[:, "Type"] == tgt_df_all.loc[:, "Type"]
        true_predicted = test.sum()
        false_predicted = (~test).sum()
        print(f"True predicted: {true_predicted}")
        print(f"False predicted: {false_predicted}")

        # maybe this is dependents on the BATCH_SIZE
        total_loss = total_loss / len(dataloader)
        return total_loss


class LitClassifier(L.LightningModule):
    def __init__(self, sequence_len: int, feature_size: int, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * (sequence_len // 4), num_classes)  # Adjust size based on pooling and conv layers

        # maybe use average = 'macro'
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_f2score = FBetaScore(task="multiclass", beta=2.0, num_classes=num_classes)
        self.train_f2score = FBetaScore(task="multiclass", beta=2.0, num_classes=num_classes)
        self.valid_f2score = FBetaScore(task="multiclass", beta=2.0, num_classes=num_classes)

    def forward(self, src: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # Reshape input to (batch_size, feature_size, sequence_len)
        src = src.permute(0, 2, 1)
        src = self.conv1(src)
        src = self.relu(src)
        src = self.pool(src)
        src = self.conv2(src)
        src = self.relu(src)
        src = self.pool(src)
        # Flatten for the fully connected layer
        src = torch.flatten(src, 1)
        src = self.fc(src)
        return src

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('train_loss', loss)
        self.log('train_f2', self.train_f2score(logits, tgt), prog_bar=True)

        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('val_loss', loss)
        self.log('val_f2', self.train_f2score(logits, tgt))
        return loss


    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy(logits, tgt))
        self.log('test_recall', self.test_recall(logits, tgt))
        self.log('test_precision', self.test_precision(logits, tgt))
        self.log('test_f2', self.test_f2score(logits, tgt))
        return loss

    def single_step(self, batch, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor, Tensor]:
        src, tgt, objectIDs, timeIndices = batch

        logits = self(src)

        logits = logits.view(logits.size(0), -1)
        tgt = tgt.reshape(-1)

        loss_fnc = nn.CrossEntropyLoss()
        loss = loss_fnc(logits, tgt)

        return loss, logits, tgt

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
