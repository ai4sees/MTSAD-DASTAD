import torch.nn as nn
import torch
import math
from torch.nn import TransformerEncoder, LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # (2 * feats, 0.1, self.n_window) #50, 0.1, 10
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (10, 50)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # [50, 25]
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 50,16
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 16 ,50
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerEncoderLayer1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer1, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # [50, 25]
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 50,16
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 16 ,50
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src2 = self.dropout(src2)
        src = self.layer_norm(src + src2)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = self.layer_norm(src + src2)

        return src


class Transformer_encoder1(nn.Module):
    def __init__(self, feats, num_heads, sequence_length, output_dim):
        super(Transformer_encoder1, self).__init__()

        self.n_feats = feats
        self.n_window = 100

        # self.learn_pos_enc = LearnablePositionalEmbedding(sequence_length, output_dim)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer1(d_model=feats, nhead=num_heads, dim_feedforward=16,
                                                  dropout=0.1)  # 50, 25
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

    def forward(self, x, training_flag=True):
        x = x * math.sqrt(self.n_feats)
        # x = self.learn_pos_enc(x, training_flag)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)

        return memory


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):  # 50
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class Conv1dLayer(nn.Module):
    def __init__(self, n_features, kernel_size=1):
        super(Conv1dLayer, self).__init__()
        self.padding1d = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv1d = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding1d(x)
        x = self.conv1d(x)
        return x.permute(0, 2, 1)  # Permute back


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back
