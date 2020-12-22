import numpy as np
import torch
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.functional import softmax, relu


"""
adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""


class GraphTransformer(torch.nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, nk_dim, ek_dim, v_dim, num_heads, ppf_hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.self_attn_layers = torch.nn.ModuleList([
            GraphSelfAttention(num_heads, node_hidden_dim, edge_hidden_dim, nk_dim, ek_dim, v_dim)
            for _ in range(num_layers)])
        self.node_pff_layers = torch.nn.ModuleList([
            PositionwiseFeedForward(node_hidden_dim, ppf_hidden_dim)
            for _ in range(num_layers)])
        self.edge_pff_layers = torch.nn.ModuleList([
            PositionwiseFeedForward(edge_hidden_dim, ppf_hidden_dim)
            for _ in range(num_layers)])

    def forward(self, node_features, edge_features, mask):
        # node_features [batch_size, num_nodes, node_dim]
        # edge_features: [batch_size, num_nodes, num_nodes, node_dim]
        # mask: [batch_size, num_nodes]
        for i in range(self.num_layers):
            node_features, edge_features, _ = self.self_attn_layers[i](node_features, edge_features, mask)
            node_features = self.node_pff_layers[i](node_features)
            edge_features = self.edge_pff_layers[i](edge_features)
        #print(torch.where(torch.isnan(node_features)))
        return node_features, edge_features


class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = Linear(d_in, d_hid)  # position-wise
        self.w_2 = Linear(d_hid, d_in)  # position-wise
        self.layer_norm = LayerNorm(d_in)
        self.dropout = Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class ScaledDotProductWithEdgeAttention(torch.nn.Module):
    def __init__(self, nk_dim, ek_dim, temperature, beta=0.5, dropout=0.1):
        super().__init__()
        self.nk_dim = nk_dim
        self.ek_dim = ek_dim
        self.temperature = temperature
        self.beta = beta
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, nk, ek, v, mask=None):
        # q:  b x nh x nn x dv
        # k:  b x nh x nn x dv
        # e:  b x nh x nn x nn x de

        # k.T:  b x nh x dv x nn
        # q x k.T --> b x nh x nn x nn
        # q x e.T: (b x nh x nn x 1 x dv) x (b x nh x nn x de x nn) --> b x nh x nn x 1 x nn

        qn, qe = torch.split(q, [self.nk_dim, self.ek_dim], dim=-1)
        node_attn = torch.matmul(qn, nk.transpose(2, 3))
        edge_attn = torch.matmul(qe.unsqueeze(3), ek.transpose(3, 4)).squeeze(-2)
        #attn = self.beta * node_attn + (1 - self.beta) * edge_attn
        attn = node_attn + edge_attn
        attn = attn / self.temperature


        # attn: b x nh x nn x nn
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # output: b x nh x nn x dv

        return output, attn


class GraphSelfAttention(torch.nn.Module):
    def __init__(self, n_head, node_hidden_dim, edge_hidden_dim, nk_dim, ek_dim, v_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.q_dim = nk_dim + ek_dim
        self.nk_dim = nk_dim
        self.ek_dim = ek_dim
        self.v_dim = v_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim

        self.w_qs = Linear(node_hidden_dim, n_head * self.q_dim, bias=False)
        self.w_nks = Linear(node_hidden_dim, n_head * self.nk_dim, bias=False)
        self.w_eks = Linear(edge_hidden_dim, n_head * self.ek_dim, bias=False)
        self.w_vs = Linear(node_hidden_dim, n_head * v_dim, bias=False)
        self.node_fc = Linear(n_head * v_dim, node_hidden_dim, bias=False)
        self.edge_fc = Linear(2 * n_head * v_dim, edge_hidden_dim, bias=False)
        self.attention = ScaledDotProductWithEdgeAttention(
            nk_dim=nk_dim,
            ek_dim=ek_dim,
            temperature=(nk_dim + ek_dim) ** 0.5
        )
        self.dropout = Dropout(dropout)
        self.layer_norm_node = LayerNorm(node_hidden_dim)
        self.layer_norm_edge = LayerNorm(edge_hidden_dim)

    def forward(self, node_emb, edge_emb, mask):
        batch_size, num_nodes = node_emb.size(0), node_emb.size(1)
        device = node_emb.device
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        node_residual = node_emb
        edge_residual = edge_emb

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x nn x nh x dv
        """q = self.w_qs(node_emb).view(batch_size, num_nodes, self.n_head, self.q_dim)
        nk = self.w_nks(node_emb).view(batch_size, num_nodes, self.n_head, self.nk_dim)
        ek = self.w_eks(edge_emb).view(batch_size, num_nodes, num_nodes, self.n_head, self.ek_dim)
        v = self.w_vs(node_emb).view(batch_size, num_nodes, self.n_head, self.v_dim)"""
        node_emb, edge_emb = node_emb[mask], edge_emb[adj_mask]
        q = torch.empty((batch_size, num_nodes, self.n_head, self.q_dim), device=device)
        nk = torch.empty((batch_size, num_nodes, self.n_head, self.nk_dim), device=device)
        ek = torch.empty((batch_size, num_nodes, num_nodes, self.n_head, self.ek_dim), device=device)
        v = torch.empty((batch_size, num_nodes, self.n_head, self.v_dim), device=device)
        q.masked_scatter_(mask[:, :, None, None], self.w_qs(node_emb))
        nk.masked_scatter_(mask[:, :, None, None], self.w_nks(node_emb))
        ek.masked_scatter_(adj_mask[:, :, :, None, None], self.w_eks(edge_emb))
        v.masked_scatter_(mask[:, :, None, None], self.w_vs(node_emb))

        # Transpose for attention dot product: b x nh x nn x dv
        q, nk, ek, v = q.transpose(1, 2), nk.transpose(1, 2), ek.transpose(1, 3), v.transpose(1, 2)

        node_emb, attn = self.attention(q, nk, ek, v, mask=adj_mask.unsqueeze(1))  # unsqueeze For head axs broadcasting

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x nn x (nh*dv)
        node_emb = node_emb.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        node_emb_combined = torch.cat(
            (node_emb.unsqueeze(1).repeat(1, num_nodes, 1, 1),
             node_emb.unsqueeze(2).repeat_interleave(num_nodes, dim=2)),
            dim=-1)  # b x nn x nn x (2*nh*dv)
        node_emb_out = torch.empty((batch_size, num_nodes, self.node_hidden_dim), device=device)
        node_emb_out.masked_scatter_(mask[:, :, None], self.dropout(self.node_fc(node_emb[mask])))
        edge_emb_out = torch.empty((batch_size, num_nodes, num_nodes, self.edge_hidden_dim), device=device)
        edge_emb_out.masked_scatter_(adj_mask[:, :, :, None], self.dropout(self.edge_fc(node_emb_combined[adj_mask])))
        node_emb_out += node_residual
        edge_emb_out += edge_residual
        node_emb_out = self.layer_norm_node(node_emb_out)
        edge_emb_out = self.layer_norm_edge(edge_emb_out)

        return node_emb_out, edge_emb_out, attn
