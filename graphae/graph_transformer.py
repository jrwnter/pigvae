import numpy as np
import torch
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.functional import softmax, relu


"""
adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""


class Transformer(torch.nn.Module):
    def __init__(self, hidden_dim, k_dim, v_dim, num_heads, ppf_hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.self_attn_layers = torch.nn.ModuleList([
            SelfAttention(num_heads, hidden_dim, k_dim, v_dim)
            for _ in range(num_layers)])
        self.pff_layers = torch.nn.ModuleList([
            PositionwiseFeedForward(hidden_dim, ppf_hidden_dim)
            for _ in range(num_layers)])

    def forward(self, x, mask):
        for i in range(self.num_layers):
            x, _ = self.self_attn_layers[i](x, mask)
            x = self.pff_layers[i](x)
        return x


class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid)  # position-wise
        self.w_2 = Linear(d_hid, d_in)  # position-wise
        self.layer_norm = LayerNorm(d_in, eps=1e-6)
        self.dropout = Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class ScaledDotProductWithEdgeAttention(torch.nn.Module):
    def __init__(self, k_dim, temperature, dropout=0.1):
        super().__init__()
        self.k_dim = k_dim
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q:  b x nh x nn x dv
        # k:  b x nh x nn x dv
        # e:  b x nh x nn x nn x de

        # k.T:  b x nh x dv x nn
        # q x k.T --> b x nh x nn x nn

        attn = torch.matmul(q, k.transpose(2, 3))
        attn = attn / self.temperature


        # attn: b x nh x nn x nn
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # output: b x nh x nn x dv

        return output, attn


class SelfAttention(torch.nn.Module):
    def __init__(self, n_head, hidden_dim, k_dim, v_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.q_dim = k_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.hidden_dim = hidden_dim

        self.w_qs = Linear(hidden_dim, n_head * self.q_dim, bias=False)
        self.w_ks = Linear(hidden_dim, n_head * self.k_dim, bias=False)
        self.w_vs = Linear(hidden_dim, n_head * v_dim, bias=False)
        self.fc = Linear(n_head * v_dim, hidden_dim, bias=False)
        self.attention = ScaledDotProductWithEdgeAttention(
            k_dim=k_dim,
            temperature=k_dim ** 0.5
        )
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, mask):
        batch_size, len_x = x.size(0), x.size(1)
        device = x.device
        interact_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        residual = x

        # Pass through the pre-attention projection: b x lx x (n*dv)
        # Separate different heads: b x lx x nh x dv
        x = x[mask]
        x = self.layer_norm(x)
        q = torch.empty((batch_size, len_x, self.n_head, self.q_dim), device=device)
        k = torch.empty((batch_size, len_x, self.n_head, self.k_dim), device=device)
        v = torch.empty((batch_size, len_x, self.n_head, self.v_dim), device=device)
        q.masked_scatter_(mask[:, :, None, None], self.w_qs(x))
        k.masked_scatter_(mask[:, :, None, None], self.w_ks(x))
        v.masked_scatter_(mask[:, :, None, None], self.w_vs(x))

        # Transpose for attention dot product: b x nh x lx x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        x, attn = self.attention(q, k, v, mask=interact_mask.unsqueeze(1))  # unsqueeze For head axs broadcasting

        # Transpose to move the head dimension back: b x lx x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lx x (nh*dv)
        x = x.transpose(1, 2).contiguous().view(batch_size, len_x, -1)
        x_out = torch.empty((batch_size, len_x, self.hidden_dim), device=device)
        x_out.masked_scatter_(mask[:, :, None], self.dropout(self.fc(x[mask])))
        x_out += residual

        return x_out, attn


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, batch_size, num_nodes):
        x = self.pos_table[:, :num_nodes].clone().detach()
        x = x.expand(batch_size, -1, -1)
        return x
