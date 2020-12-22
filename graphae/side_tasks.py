import torch
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.functional import relu


class PropertyPredictor(torch.nn.Module):

    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid)
        self.w_2 = Linear(d_hid, d_hid)
        self.w_3 = Linear(d_hid, d_out)
        self.layer_norm1 = LayerNorm(d_hid)
        self.layer_norm2 = LayerNorm(d_hid)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm1(self.dropout(relu(self.w_1(x))))
        x = self.layer_norm2(self.dropout(relu(self.w_2(x))))
        x = self.w_3(x)
        return x