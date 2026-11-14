
import math
import torch
import torch.nn as nn

# ============================================================
# 1. MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask.bool(),
                torch.finfo(scores.dtype).min
            )

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, V)

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size, -1, self.d_model
        )

        return self.Wo(output)


# ============================================================
# 2. POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds positional encoding length.")

        return x + self.pe[:, :x.size(1)]


# ============================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ============================================================

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):

        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


# ============================================================
# 4. ENCODER LAYER
# ============================================================

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):

        attention_output = self.self_attention(
            x, x, x, src_mask
        )

        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout2(ff_output)
        )

        return x


# ============================================================
# 5. DECODER LAYER
# ============================================================

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout
        )

        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask,
        src_mask
    ):

        attention_output = self.self_attention(
            x, x, x, tgt_mask
        )

        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        cross_output = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            src_mask
        )

        x = self.norm2(
            x + self.dropout2(cross_output)
        )

        ff_output = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout3(ff_output)
        )

        return x


# ============================================================
# 6. NEXUS-NMT TRANSFORMER
# ============================================================

class NexusNMT(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        pad_id=0
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=pad_id
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=pad_id
        )

        self.positional_encoding = PositionalEncoding(
            d_model
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    # --------------------------------------------------------
    # ATTENTION MASKS
    # --------------------------------------------------------

    def generate_mask(self, src, tgt):

        # Source key padding mask: [B, 1, 1, S]
        src_mask = (
            src != self.pad_id
        ).unsqueeze(1).unsqueeze(2)

        # Target key padding mask: [B, 1, 1, T]
        tgt_padding_mask = (
            tgt != self.pad_id
        ).unsqueeze(1).unsqueeze(2)

        target_length = tgt.size(1)

        # Causal mask: [1, 1, T, T]
        causal_mask = torch.tril(
            torch.ones(
                target_length,
                target_length,
                device=tgt.device,
                dtype=torch.bool
            )
        ).unsqueeze(0).unsqueeze(0)

        tgt_mask = tgt_padding_mask & causal_mask

        return src_mask, tgt_mask

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    def forward(self, src, tgt):

        src_mask, tgt_mask = self.generate_mask(
            src, tgt
        )

        # Encoder
        src_embedding = self.src_embedding(src)
        src_embedding = src_embedding * math.sqrt(self.d_model)

        encoder_output = self.positional_encoding(
            src_embedding
        )

        encoder_output = self.dropout(encoder_output)

        for layer in self.encoder_layers:

            encoder_output = layer(
                encoder_output,
                src_mask
            )

        # Decoder
        tgt_embedding = self.tgt_embedding(tgt)
        tgt_embedding = tgt_embedding * math.sqrt(self.d_model)

        decoder_output = self.positional_encoding(
            tgt_embedding
        )

        decoder_output = self.dropout(decoder_output)

        for layer in self.decoder_layers:

            decoder_output = layer(
                decoder_output,
                encoder_output,
                tgt_mask,
                src_mask
            )

        logits = self.output_projection(decoder_output)

        return logits


print("NexusNMT architecture defined successfully.")
