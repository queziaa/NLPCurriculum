"""
Model code has been adapted from https://nlp.seas.harvard.edu/annotated-transformer/
The original code was modified to suit the requirements of this project.
Credit goes to the original author for their contribution.

All figure/equation references refer to "Attention is All You Need" paper
"""


import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, adapter=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.adapter = adapter

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        src = self.encode(src, src_mask)
        src = self.adapter(src)
        tgt_decoded = self.decode(src, src_mask, tgt, tgt_mask)
        scores = self.generator(tgt_decoded)

        return scores

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear transformation + softmax to convert the decoder output to next-token probabilities."""
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, level=None):
        return F.log_softmax(self.proj(x), dim=-1)


class Adapter(nn.Module):
    """Define standard linear transformation, from source dim to target dim."""
    def __init__(self, d_src, d_tgt):
        super(Adapter, self).__init__()
        self.proj = nn.Linear(d_src, d_tgt)

    def forward(self, x):
        y = self.proj(x)
        return y


def clones(module, N):
    """Produce N identical layers (for encoder/decoder)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers."""
    def __init__(self, layer, N, convolver=None):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)."""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # for residual connections
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # for residual connections

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """Same as torch.nn.LayerNorm."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2."""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN (Equation 2) for encoder/decoder layers."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, emb_init=None, need_training=True):
        super(Embeddings, self).__init__()

        if emb_init is not None:
            emb_weight = torch.from_numpy(emb_init).float()
            self.lut = nn.Embedding.from_pretrained(
                emb_weight,
                padding_idx=padding_idx,
                sparse=True
            )
        else:
            self.lut = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx,
                sparse=True
            )

        self.lut.weight.requires_grad = need_training
        self.d_model = embedding_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Encode information about the position of the tokens in the sequence.
    Sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(vocab_src, vocab_tgt, config, emb_src_init=None, emb_tgt_init=None):
    """
    Create model instance
    :param vocab_src: instance of torchtext.vocab.Vocab, mapping between tokens and their ids
    :param vocab_tgt: instance of torchtext.vocab.Vocab, mapping between labels and their ids
    :param config: config
    :param emb_src_init: initialized token embeddings; torch.tensor (src_vocab_size x d_src)
    :param emb_tgt_init: initialized label embeddings; torch.tensor (tgt_vocab_size x d_tgt)
    :return: EncoderDecoder instance
    """
    N_src = config["Model"].getint("layers_src")
    N_tgt = config["Model"].getint("layers_tgt")
    d_src = config["Model"].getint("d_src")
    d_tgt = config["Model"].getint("d_tgt")
    d_ff = config["Model"].getint("d_ff")
    h = config["Model"].getint("heads")
    dropout = config["Model"].getfloat("dropout")

    c = copy.deepcopy
    encoder_attn = MultiHeadedAttention(h, d_src)
    encoder_ff = PositionwiseFeedForward(d_src, d_ff, dropout)
    decoder_attn = MultiHeadedAttention(h, d_tgt)
    decoder_ff = PositionwiseFeedForward(d_tgt, d_ff, dropout)
    position = PositionalEncoding(d_src, dropout)

    encoder = Encoder(
        layer=EncoderLayer(
            size=d_src,
            self_attn=c(encoder_attn),
            feed_forward=c(encoder_ff),
            dropout=dropout
        ),
        N=N_src
    )
    decoder = Decoder(
        layer=DecoderLayer(
            size=d_tgt,
            self_attn=c(decoder_attn),
            src_attn=c(decoder_attn),
            feed_forward=c(decoder_ff),
            dropout=dropout
        ),
        N=N_tgt
    )

    src_embed = Embeddings(
        vocab_size=len(vocab_src),
        embedding_dim=d_src,
        padding_idx=vocab_src["<blank>"],
        emb_init=emb_src_init,
        need_training=True
    )

    tgt_embed = Embeddings(
        vocab_size=len(vocab_tgt),
        embedding_dim=d_tgt,
        padding_idx=vocab_tgt["<blank>"],
        emb_init=emb_tgt_init,
        need_training=True
    )

    generator = Generator(
        d_model=d_tgt,
        vocab_size=len(vocab_tgt),
    )

    adapter = Adapter(d_src, d_tgt)

    model = EncoderDecoder(
        encoder,
        decoder,
        src_embed=nn.Sequential(src_embed, c(position)),
        tgt_embed=tgt_embed,
        generator=generator,
        adapter=adapter
    )

    print("Initializing weights")
    for name, param in model.named_parameters():
        if name == "src_embed.0.lut.weight" and emb_src_init is not None:
            print(f"skipped {name}")
        elif name == "tgt_embed.lut.weight" and emb_tgt_init is not None:
            print(f"skipped {name}")
        elif param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model


def load_trained_model(config, model_path, vocab_src, vocab_tgt):
    """
    Load pretrained model from model_path
    :param config: config
    :param model_path: path to the model
    :param vocab_src: instance of torchtext.vocab.Vocab, mapping between tokens and their ids
    :param vocab_tgt: instance of torchtext.vocab.Vocab, mapping between labels and their ids
    :return: EncoderDecoder instance
    """
    if not os.path.exists(model_path):
        raise Exception(f"{model_path}: model not found")

    model = make_model(vocab_src, vocab_tgt, config)

    checkpoint = torch.load(model_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.cuda()

    return model
