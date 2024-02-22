from pathlib import Path
from typing import Optional, List, Callable

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from timeit import default_timer as timer

from baseline_submissions.evaluation import NodeDetectionEvaluator
from baseline_submissions.own_model.dataset_manip import convert_tgts_for_eval, MyDataset, state_change_eval


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
    def __init__(self, src_size: int, emb_size: int, tgt_size: int, nhead: int, d_hid: int = 2048, nlayers: int = 2,
                 dropout: float = 0.5, batch_first: bool = True):
        super().__init__()

        # layer stuff
        self.normalisation = nn.LayerNorm(src_size, elementwise_affine=True)  # Learning params for now?
        self.pos_encoder = PositionalEncoding(src_size, dropout)
        self.linear_in = nn.Linear(src_size, emb_size)

        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, d_hid, dropout, batch_first=batch_first,
                                                   activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.linear_out = nn.Linear(emb_size, tgt_size)

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
        src = self.linear_in(self.pos_encoder(self.normalisation(src)))
        # TUTORIAL HAS THIS. DO I REALLY NEED THIS???? FOR SOURCE IT SHOULDNT MATTER WHAT IT SEES RIGHT????
        # if src_mask is None:
        #     """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        #     Unmasked positions are filled with float(0.0).
        #     """
        #     src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.encoder(src, src_mask, src_padding_mask)
        output = self.linear_out(output)
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

        for src, tgt, objectIds in dataloader:
            src_mask = create_src_mask(src, self.nhead, device)
            src_padding_mask = create_src_padding_mask(src, padding_vec, device)

            src = src.to(device)
            tgt = tgt.to(device)

            pred = self.forward(src, src_mask, src_padding_mask)
            optimizer.zero_grad()

            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
            loss.backward()

            # IDK if I need this?
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
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

        for src, tgt, objectIDs in dataloader:
            src_mask = create_src_mask(src, self.nhead, device)
            src_padding_mask = create_src_padding_mask(src, padding_vec, device)

            src = src.to(device)
            tgt = tgt.to(device)

            pred = self.forward(src, src_mask, src_padding_mask)
            loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
            total_loss += loss.item()

            pred_df, tgt_df = convert_tgts_for_eval(pred, tgt, objectIDs, tgt_dict)
            pred_df_all.append(pred_df)
            tgt_df_all.append(tgt_df)

        pred_df_all = pd.concat(pred_df_all)
        tgt_df_all = pd.concat(tgt_df_all)

        pred_df_all.to_csv(Path("./evaluations/pred.csv"), index=False)
        tgt_df_all.to_csv(Path("./evaluations/tgt.csv"), index=False)
        print("Evaluation stored")

        evaluator = NodeDetectionEvaluator(tgt_df_all, pred_df_all, 6)

        # maybe this is dependents on the BATCH_SIZE
        total_loss = total_loss / len(dataloader)
        return evaluator, total_loss


class TimeSeriesCNN(nn.Module):
    def __init__(self, feature_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, 1)  # Output layer for binary classification

    def forward(self, x):
        # Assuming x is of shape (batch_size, sequence_length, feature_size)
        # Conv1d expects (batch_size, in_channels, sequence_length)
        x = F.normalize(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Apply global average pooling to aggregate features across the temporal dimension
        # x = F.adaptive_avg_pool1d(x, 1)  # Uncomment if pooling is desired
        x = x.permute(0, 2, 1)  # Rearrange dimensions for the linear layer
        x = self.fc(x)
        x = torch.sigmoid(x)  # Sigmoid activation to output probabilities
        return x.squeeze(-1)  # Removing the last dimension to match target shape

    def do_train(self, dataloader: DataLoader, num_epochs: int, optimizer: torch.optim.Optimizer, loss_fn: Callable,
                 device: torch.device, store_model: bool = True,
                 trained_model_path: Path = None) -> List[float]:

        loss = []
        print("Starting Training: ---------------------------------")
        for epoch in range(1, num_epochs + 1):
            start_time = timer()
            train_loss = self._train_epoch(dataloader, optimizer, loss_fn, device)
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
                     device: torch.device):
        self.train()
        losses = 0.0

        for src, tgt, objectIds in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            pred = self.forward(src)
            optimizer.zero_grad()

            loss = loss_fn(pred.reshape(-1), tgt.reshape(-1))
            loss.backward()

            # IDK if I need this?
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            losses += loss.item()

        # maybe this is dependents on the BATCH_SIZE
        return losses / len(dataloader)

    @torch.no_grad()
    def do_test(self, dataloader: DataLoader, loss_fn: Callable, device: torch.device):
        self.eval()

        total_loss = 0.0
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0

        for src, tgt, objectIDs in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            pred = self.forward(src)
            loss = loss_fn(pred.reshape(-1), tgt.reshape(-1))
            total_loss += loss.item()

            m = nn.Sigmoid()
            pred = m(pred)
            TP, FP, TN, FN = state_change_eval(pred, tgt)
            total_tp += TP
            total_fp += FP
            total_tn += TN
            total_fn += FN

        print(f"Total TPs: {total_tp}")
        print(f"Total FPs: {total_fp}")
        print(f"Total FNs: {total_fn}")

        precision = total_tp / (total_tp + total_fp) \
            if (total_tp + total_fp) != 0 else 0
        recall = total_tp / (total_tp + total_fn) \
            if (total_tp + total_fn) != 0 else 0
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
            if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0

        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2: {f2:.2f}')

        # maybe this is dependents on the BATCH_SIZE
        total_loss = total_loss / len(dataloader)
        return total_loss
