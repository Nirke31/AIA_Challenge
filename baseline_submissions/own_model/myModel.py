import torch
from torch import Tensor
import torch.nn as nn
import math


def create_mask(src: torch.Tensor, tgt: torch.Tensor, num_heads: int, src_pad_vector: torch.Tensor,
                tgt_pad_vector: torch.Tensor, device: torch.device):
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
    Returns: src, tgt, src padding, tgt padding and memory masks
    """
    batch_len = src.shape[0]
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    # In our case src and tgt seq len should be the same but let's not assume
    src_num_features = src_pad_vector.shape[0]
    tgt_num_features = tgt_pad_vector.shape[0]

    # src and tgt mask, masking the src and tgt sequence. Has to be of shape: batch len * NHEAD, seq len, seq len
    # I think I do not have to add the padding here because already in padding_mask?
    src_mask = torch.zeros((batch_len * num_heads, src_seq_len, src_seq_len), device=device).type(torch.bool)
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
    tgt_mask = tgt_mask.unsqueeze(0).repeat(batch_len * num_heads, 1, 1)

    # Create padding mask (num features, sequence length). True where padding vector
    src_padding_mask = (src == src_pad_vector.view(1, 1, src_num_features)).all(dim=2)
    tgt_padding_mask = (tgt == tgt_pad_vector.view(1, 1, tgt_num_features)).all(dim=2)

    # Create memory mask, idk what it does yet so just zeros(mask is irrelevant)? example used same as source
    memory_mask = torch.zeros((tgt_seq_len, src_seq_len))
    memory_padding_mask = src_padding_mask
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_mask, memory_padding_mask


# Copied from pytorch. I am unsure why we are not directly using the embedding by multiplying it with the squareroot
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings.
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
# input are just our features, tgt is the sequence of strings, we have to encode tgt such that one string combination
# represents one vector. That vector has to be same length as features.
class Seq2SeqTransformer(nn.Module):
    """
    Note: batch_first = True
    """
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 # emb_size: int,  # for now we don't use this and just ust the src_size
                 nhead: int,
                 src_size: int,
                 tgt_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=src_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        self.tgt_emb = TokenEmbedding(tgt_size,
                                      src_size)  # here we map from target to source_size/feature size cuz needed for transformer
        self.generator = nn.Linear(src_size, tgt_size)

    def forward(self, src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                memory_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_padding_mask: Tensor):

        tgt_emb = self.tgt_emb(tgt)
        outs = self.transformer(src, tgt_emb, src_mask, tgt_mask, memory_mask, src_padding_mask,
                                tgt_padding_mask, memory_padding_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(tgt, memory, tgt_mask)
