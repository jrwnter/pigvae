import torch
from torch.nn import Linear, Sequential, ELU, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GATConv, NNConv, GINConv, global_mean_pool


class EdgeNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, attr_dim):
        super().__init__()
        nn = Sequential(
            Linear(attr_dim, 256),
            ELU(),
            Linear(256, 1024),
            ELU(),
            Linear(1024, in_channels * out_channels),
            ELU())
        self.conv = NNConv(in_channels, out_channels, nn)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, num_atom_features, heads=1, batch_norm=False,
                 weight_sharing=False):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.edge_conv = EdgeNNConv(num_atom_features, hidden_dim, 5)
        conv_layers = [GATConv(hidden_dim, hidden_dim, heads)]
        if weight_sharing:
            conv_layers += (num_layers - 2) * [GATConv(heads * hidden_dim, hidden_dim, heads)]
        else:
            conv_layers += [GATConv(heads * hidden_dim, hidden_dim, heads) for _ in range(num_layers - 2)]
        conv_layers += [GATConv(heads * hidden_dim, node_dim, heads, concat=False)]
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        if batch_norm:
            bn_layers = [BatchNorm1d(hidden_dim)]
            bn_layers += [BatchNorm1d(heads * hidden_dim) for _ in range(num_layers - 1)]
            self.bn_layers = torch.nn.ModuleList(bn_layers)

    def forward(self, x, edge_index, edge_attr):
        x = self.edge_conv(x, edge_index, edge_attr)
        for i in range(self.num_layers):
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.conv_layers[i](x, edge_index)
        return x


class WholeGraphEncoder(torch.nn.Module):
    def __init__(self, node_dim, emb_dim, hidden_dim):
        super().__init__()
        self.fc1 = Linear(node_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, hidden_dim)
        self.fc4 = Linear(hidden_dim, emb_dim)
        self.bn1 = BatchNorm1d(node_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim)

    def forward(self, node_emb, batch):
        # TODO: Put batchnorm on this input node?
        x = node_emb
        x = F.elu(self.bn1(x))
        x = F.elu(self.bn2(self.fc1(x)))
        x = F.elu(self.bn3(self.fc2(x)))
        x = global_mean_pool(x, batch)
        x = F.elu(self.bn4(self.fc3(x)))
        x = self.fc4(x)
        return x
