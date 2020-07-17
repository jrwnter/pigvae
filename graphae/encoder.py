import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import DenseGCNConv
from graphae.fully_connected import FNN


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, batch_norm=False, non_linearity="relu",
                 weight_sharing=False):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        conv_layers = [DenseGCNConv(input_dim, hidden_dim)]
        if weight_sharing:
            conv_layers += (num_layers - 2) * [DenseGCNConv(hidden_dim, hidden_dim)]
        else:
            conv_layers += [DenseGCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        conv_layers += [DenseGCNConv(hidden_dim, output_dim)]
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        if batch_norm:
            bn_layers = [BatchNorm1d(hidden_dim) for _ in range(num_layers)]
            self.bn_layers = torch.nn.ModuleList(bn_layers)
        if non_linearity == "relu":
            self.non_linearity = torch.nn.ReLU()
        elif non_linearity == "elu":
            self.non_linearity = torch.nn.ELU()

    def forward(self, x, adj, mask=None):
        for i in range(self.num_layers):
            if i > 0:
                x = self.non_linearity(x)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.conv_layers[i](x, adj, mask)
        return x


class NodeAggregator(torch.nn.Module):
    def __init__(self, node_dim, emb_dim, hidden_dim, num_layers, num_nodes, batch_norm=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.fnn = FNN(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            num_layers=num_layers,
            non_linearity="elu",
            batch_norm=batch_norm
        )

    def forward(self, node_emb, mask):
        x = torch.mean(node_emb.view(-1, self.num_nodes, self.node_dim), dim=1)
        # TODO: put activation func here?
        x = self.fnn(x)
        return x

