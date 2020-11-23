import torch
from torch.nn import Linear, BatchNorm1d
from graphae.graph_transformer import TransformerConv


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, edge_dim, non_lin, batch_norm):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.layers = [TransformerConv(input_dim, hidden_dim, num_heads, edge_dim=edge_dim)]
        self.layers += [TransformerConv(num_heads * hidden_dim, hidden_dim, num_heads, edge_dim=edge_dim) for _ in range(num_layers - 1)]
        self.layers = torch.nn.ModuleList(self.layers)
        self.linear_out = Linear(num_heads * hidden_dim, output_dim)
        if batch_norm:
            bn_layers = [BatchNorm1d(num_heads * hidden_dim) for _ in range(num_layers)]
            bn_layers += [BatchNorm1d(output_dim)]
            self.bn_layers = torch.nn.ModuleList(bn_layers)
        if non_lin == "relu":
            self.non_lin = torch.nn.ReLU()
        elif non_lin == "elu":
            self.non_lin = torch.nn.ELU()
        elif non_lin == "lrelu":
            self.non_lin = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        # x [batch_size, node_dim]
        # edge_index: dense edge_index (all combinations of num_nodes)
        mask = (torch.arange(edge_index.size(1)) % 2).bool()
        for i in range(self.num_layers):
            if i % 2 == 0:
                ei = edge_index[:, ~mask]
                ea = edge_attr[~mask]
            else:
                ei = edge_index[:, mask]
                ea = edge_attr[mask]
            x = self.layers[i](x, ei, ea)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.non_lin(x)
        x = self.linear_out(x)
        if self.batch_norm:
            x = self.bn_layers[-1](x)
        x = self.non_lin(x)
        return x