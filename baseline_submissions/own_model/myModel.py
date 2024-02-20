from typing import Optional, List

import torch
from torch import Tensor
import torch.nn as nn
import math


def create_mask(src: torch.Tensor, tgt: torch.Tensor, num_heads: int, src_pad_vector: torch.Tensor,
                tgt_pad_vector: torch.Tensor, device: torch.device, which_masks: List[bool]):
    """
    Creates masks needed for transformer. Transformer docu shows dimensions of masks. Let's just create all mask even
    if we do not need them.
    Args:
        src: src data, format: batch size, source sequence, source features
        tgt: tgt data
        num_heads: num attention heads, needed for src and target masks
        src_pad_vector: src padding vector. Used for src_padding_mask
        tgt_pad_vector: tgt padding vector. Used for tgt_padding_mask
        device: DEVICE
        which_masks: List of bools, size 6, True or False which mask should be returned. Order is: src, tgt, memory,
        src_padding, tgt_padding, memory_padding. Same as return order.
    Returns: ORDER IS IMPORTANT RN, src, tgt, memory masks as well as padding masks
    """
    batch_len = src.shape[0]
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    # In our case src and tgt seq len should be the same but let's not assume
    src_num_features = src_pad_vector.shape[0]
    tgt_num_features = tgt_pad_vector.shape[0]

    if len(which_masks) != 6: raise ValueError("List of Size 6 expected")

    src_mask = None
    tgt_mask = None
    memory_mask = None
    src_padding_mask = None
    tgt_padding_mask = None
    memory_padding_mask = None

    # src and tgt mask, masking the src and tgt sequence. Has to be of shape: batch len * NHEAD, seq len, seq len
    # I think I do not have to add the padding here because already in padding_mask?
    if which_masks[0]:
        src_mask = torch.zeros((batch_len * num_heads, src_seq_len, src_seq_len), device=device).type(torch.bool)
    if which_masks[1]:
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
        tgt_mask = tgt_mask.unsqueeze(0).repeat(batch_len * num_heads, 1, 1)

    # Create padding mask (num features, sequence length). True where padding vector
    if which_masks[2]:
        src_padding_mask = (src == src_pad_vector.view(1, 1, src_num_features)).all(dim=2).to(device=device)
    if which_masks[3]:
        tgt_padding_mask = (tgt == tgt_pad_vector.view(1, 1, tgt_num_features)).all(dim=2).to(device=device)

    # Create memory mask, idk what it does yet so just zeros(mask is irrelevant)? example used same as source
    if which_masks[4]:
        memory_mask = torch.zeros((tgt_seq_len, src_seq_len), device=device).type(torch.bool)
    if which_masks[5]:
        memory_padding_mask = src_padding_mask

    return src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, memory_padding_mask


# Copied from pytorch. I am unsure why we are not directly using the embedding by multiplying it with the squareroot.
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings.
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=34)  # TODO: GET THIS FROM SOMEWHERE
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # We have to remove the feature dimension because the torch.emb doesn't want it
        return self.embedding(tokens.squeeze(dim=-1)) * math.sqrt(self.emb_size)


# Seq2Seq Network
# input are just our features, tgt is the sequence of strings, we have to encode tgt such that one string combination
# represents one vector. That vector has to be same length as features.
class Seq2SeqTransformer(nn.Module):
    """
    Input are just our src features, tgt is the sequence of strings. We have to encode tgt such that one
    string combination represents one vector. That vector has to be same length as features.
    Note: batch_first = True
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 nhead: int,
                 src_size: int,  # this is the 'embedding' size I think?
                 tgt_size: int,  # number of targets
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.src_normalisation = nn.LayerNorm(src_size)
        self.transformer = nn.Transformer(d_model=src_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        # here we map from target to source_(feature)size cuz needed for transformer
        self.tgt_emb = TokenEmbedding(tgt_size, src_size)
        self.generator = nn.Linear(src_size, tgt_size)

    def forward(self, src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None,
                memory_padding_mask: Optional[Tensor] = None):
        # normalise src and get tgt embedding
        src = self.src_normalisation(src)
        tgt_emb = self.tgt_emb(tgt)

        # actual transformer model
        outs = self.transformer(src, tgt_emb, src_mask, tgt_mask, memory_mask, src_padding_mask,
                                tgt_padding_mask, memory_padding_mask)

        # output
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.tgt_emb(tgt), memory, tgt_mask)
