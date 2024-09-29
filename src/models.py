import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *


class DASTAD(nn.Module):
    def __init__(self, feats):
        super(DASTAD, self).__init__()
        self.name = 'DASTAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.n_head_f = 19
        self.n_head_t = 2
        self.kernel_size = 7
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)  # 50, 0.1, 10
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16,
                                                 dropout=0.1)  # 50, 25
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())  # 50, 25
        # self.preprocess = MTAD_GAT(self.n_feats, self.n_window, 19, 2)

        self.conv = ConvLayer(feats, self.kernel_size)
        self.transformer_encoder_f1 = Transformer_encoder1(feats, self.n_head_f, sequence_length=self.n_window,
                                                           output_dim=feats)
        self.transformer_encoder_f5 = Transformer_encoder1(3 * feats, self.n_head_f * 3, sequence_length=self.n_window,
                                                           output_dim=3 * feats)
        self.transformer_encoder_t1 = Transformer_encoder1(self.n_window, self.n_head_t, sequence_length=feats,
                                                           output_dim=self.n_window)

        self.conv1d = Conv1dLayer(3 * feats, 1)
        self.project = nn.Linear(3 * feats, feats)

    def preprocess(self, x, training_flag=True):
        x = x.permute(1, 0, 2)
        conv_x = self.conv(x)

        inp1 = conv_x.permute(1, 0, 2)
        inp2 = conv_x.permute(2, 0, 1)
        h_feat1 = self.transformer_encoder_f1(inp1, training_flag)
        h_temp1 = self.transformer_encoder_t1(inp2, training_flag)
        h_feat = h_feat1.permute(1, 0, 2)
        h_temp = h_temp1.permute(1, 2, 0)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        h_cat = self.conv1d(h_cat)

        h_cat = h_cat.permute(1, 0, 2)
        h_cat = self.transformer_encoder_f5(h_cat, training_flag)
        h_end = h_cat.permute(1, 0, 2)

        h_end = self.project(h_end)
        h_end = h_end.permute(1, 0, 2)

        return h_end

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt, training_flag=True):
        src = self.preprocess(src, training_flag)

        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2
