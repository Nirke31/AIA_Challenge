from pathlib import Path
from typing import Optional, List, Callable, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from timeit import default_timer as timer
import lightning as L
from torchmetrics.classification import Accuracy, FBetaScore, Recall, Precision, BinaryPrecisionRecallCurve
from torchmetrics import MetricCollection
from own_model.src.multiScale1DResNet import MSResNet, SimpleResNet, DumbNet


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
    def __init__(self, sequence_len: int, feature_size: int, learning_rate: float, num_classes: int):
        super().__init__()
        # needed for loading model
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # self.net = DumbNet(feature_size, sequence_len, num_classes)

        # # we add one in _prepare_source
        # feature_size += 1

        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=16, kernel_size=13, stride=1, padding=6)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(64 * (sequence_len // 8), 1024)  # Adjust size based on pooling and conv layers
        self.fc1 = nn.Linear(1024, num_classes)  # Adjust size based on pooling and conv layers

        # maybe use average = 'macro'
        # self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        # self.test_recall = Recall(task="multiclass", num_classes=num_classes)
        # self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        # self.test_f2score = FBetaScore(task="multiclass", beta=2.0, num_classes=num_classes)
        # self.train_f2score = FBetaScore(task="multiclass", beta=2.0, num_classes=num_classes)
        # self.valid_f2score = FBetaScore(task="multiclass", beta=2.0, num_classes=num_classes)

        metric = MetricCollection([Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
                                   Recall(task="multiclass", num_classes=num_classes, average="macro"),
                                   Precision(task="multiclass", num_classes=num_classes, average="macro"),
                                   FBetaScore(task="multiclass", num_classes=num_classes, average="macro", beta=2.0)])
        self.train_metrics = metric.clone(prefix="train_")
        self.val_metrics = metric.clone(prefix="val_")
        self.test_metrics = metric.clone(prefix="test_")

        # Define tracker over the collection to easy keep track of the metrics over multiple epochs
        self.train_tracker = torchmetrics.wrappers.MetricTracker(self.train_metrics)
        self.val_tracker = torchmetrics.wrappers.MetricTracker(self.val_metrics)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2,
                                                               patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, src: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # Reshape input to (batch_size, feature_size, sequence_len)

        src = src.permute(0, 2, 1)

        src = self.conv1(src)
        src = self.relu(src)
        src = self.pool(src)

        src = self.conv2(src)
        src = self.relu(src)
        src = self.pool(src)

        src = self.conv3(src)
        src = self.relu(src)
        src = self.pool(src)
        # Flatten for the fully connected layer
        src = torch.flatten(src, 1)
        src = self.fc(src)
        src = self.relu(src)
        src = self.fc1(src)
        return src

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('train_loss', loss)
        metrics = self.train_metrics(logits, tgt)
        self.log_dict(metrics, prog_bar=True)

        # tracker for plots after training finished
        self.train_tracker.update(logits, tgt)

        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('val_loss', loss)
        metrics = self.val_metrics(logits, tgt)
        self.log_dict(metrics, prog_bar=False)

        # tracker for plots after training finished
        self.val_tracker.update(logits, tgt)

        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        metrics = self.test_metrics(logits, tgt)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def single_step(self, batch, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor, Tensor]:
        src, tgt, objectIDs, timeIndices = batch

        logits = self(src)

        logits = logits.view(logits.size(0), -1)
        tgt = tgt.reshape(-1)

        loss_fnc = nn.CrossEntropyLoss()
        loss = loss_fnc(logits, tgt)

        return loss, logits, tgt

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        src, objectIDs, timeSteps = batch
        logits = self(src)
        logits = logits.view(logits.size(0), -1)
        pred = torch.argmax(logits, dim=1)
        return pred, objectIDs, timeSteps

    def on_train_epoch_start(self) -> None:
        # initialize a new metric for being tracked - for this epoch I think
        self.train_tracker.increment()

    def on_validation_epoch_start(self) -> None:
        # initialize a new metric for being tracked - for this epoch I think
        self.val_tracker.increment()

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output, prog_bar=False)
        # reset for early stop
        self.val_metrics.reset()

    def on_test_end(self) -> None:
        # train and val trackers
        all_train_results = self.train_tracker.compute_all()
        all_val_results = self.val_tracker.compute_all()
        self.train_tracker.plot(val=all_train_results)
        self.val_tracker.plot(val=all_val_results)

        # plt.show()
        self.val_metrics.reset()


class LitChangePointClassifier(L.LightningModule):
    def __init__(self, feature_size: int, seq_len: int):
        super().__init__()
        # needed for loading model
        self.save_hyperparameters()

        self.model = DumbNet(feature_size, seq_len)

        # metrics
        metric = MetricCollection([Accuracy(task="binary"),
                                   Recall(task="binary"),
                                   Precision(task="binary"),
                                   FBetaScore(task="binary", beta=2.0)])
        self.train_metrics = metric.clone(prefix="train_")
        self.val_metrics = metric.clone(prefix="val_")
        self.test_metrics = metric.clone(prefix="test_")
        # extra because returning value pairs and not a single scalar
        self.test_BPRC = BinaryPrecisionRecallCurve()

        # Define tracker over the collection to easy keep track of the metrics over multiple epochs
        self.train_tracker = torchmetrics.wrappers.MetricTracker(self.train_metrics)
        self.val_tracker = torchmetrics.wrappers.MetricTracker(self.val_metrics)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def forward(self, src: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # Reshape input to (batch_size, feature_size, sequence_len)
        src = src.permute(0, 2, 1)
        out = self.model(src)
        return out

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('train_loss', loss)
        metrics = self.train_metrics(logits, tgt)
        self.log_dict(metrics, prog_bar=True)

        # tracker for plots after training finished
        self.train_tracker.update(logits, tgt)

        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        self.log('val_loss', loss)
        metrics = self.val_metrics(logits, tgt)
        self.log_dict(metrics)

        # tracker for plots after training finished
        self.val_tracker.update(logits, tgt)

        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        loss, logits, tgt = self.single_step(batch)

        # log stuff
        metrics = self.test_metrics(logits, tgt)
        self.test_BPRC.update(logits, tgt.to(dtype=torch.int))
        self.log_dict(metrics, prog_bar=True)

        return loss

    def single_step(self, batch, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor, Tensor]:
        src, tgt = batch

        logits = self(src)

        logits = logits.view(logits.size(0), -1)
        tgt = tgt.squeeze(-1)

        weight = Tensor([3]).cuda()
        loss_fnc = nn.BCEWithLogitsLoss(pos_weight=weight)  # Look at this again, num_neg/num_pos
        loss = loss_fnc(logits, tgt)

        return loss, logits, tgt

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        src, objectIDs, timeSteps = batch
        logits = self(src)
        logits = logits.view(logits.size(0), -1)
        pred = torch.argmax(logits, dim=1)
        return pred, objectIDs, timeSteps

    def on_train_epoch_start(self) -> None:
        # initialize a new metric for being tracked - for this epoch I think
        self.train_tracker.increment()

    def on_validation_epoch_start(self) -> None:
        # initialize a new metric for being tracked - for this epoch I think
        self.val_tracker.increment()

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()

    def on_test_end(self) -> None:
        # Plot Binary precision recall curve
        self.test_BPRC.plot()

        # train and val trackers
        all_train_results = self.train_tracker.compute_all()
        all_val_results = self.val_tracker.compute_all()
        self.train_tracker.plot(val=all_train_results)
        self.val_tracker.plot(val=all_val_results)

        plt.show()
        self.val_metrics.reset()


class Autoencoder(L.LightningModule):
    def __init__(self, num_input: int, num_hid: int, num_bottleneck: int, act_fn: Callable = nn.GELU):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, num_hid),
            act_fn(),  # inplace=True, do I need this?
            nn.Linear(num_hid, num_bottleneck),
            act_fn()
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_bottleneck, num_hid),
            act_fn(),
            nn.Linear(num_hid, num_input),
            nn.Tanh()
        )
        # its just ez this way
        self.loss_epoch = []
        self.loss = []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _get_reconstruction_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1]).mean(dim=[0])
        return loss

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _single_step(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        # the TensorDataset returns a list with my batch for some fkin reason
        x = batch[0]
        x_hat = self.forward(x)
        loss = self._get_reconstruction_loss(x, x_hat)
        return x, x_hat, loss

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, x_hat, loss = self._single_step(batch)
        self.log("training_loss", loss, prog_bar=True)
        self.loss_epoch.append(loss.detach().cpu())
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, x_hat, loss = self._single_step(batch)
        self.log("val_loss", loss)
        return

    def encode_features(self, data):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(data)

        return encoded

    def on_train_epoch_end(self) -> None:
        self.loss.append(np.mean(self.loss_epoch))
        self.loss_epoch = []

    def on_train_end(self) -> None:
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.show()
