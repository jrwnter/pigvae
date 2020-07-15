import torch
from torch.nn import Linear, Sequential, ELU, BatchNorm1d, ReLU
from graph_ae.modules.fully_connected import FNN


class MetaEdgeDecoder(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, meta_node_dim, hidden_dim, num_layers, batch_norm=False):
        super().__init__()
        self.num_nodes = num_nodes,
        self.meta_node_dim = meta_node_dim
        self.fnn = FNN(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * meta_node_dim,
            num_layers=num_layers,
            non_linearity="elu",
            batch_norm=batch_norm
        )

    def forward(self, x):
        x = self.fnn(x)
        x = x.view(-1, self.num_nodes, self.meta_node_dim)
        return x


class EdgePredictor(torch.nn.Module):
    def __init__(self, num_nodes, meta_node_dim, hidden_dim, num_layers, batch_norm=False):
        super().__init__()
        self.num_nodes = num_nodes,
        self.meta_node_dim = meta_node_dim
        self.output_dim = 1
        self.idxs = self.get_combination_idxs()
        self.fnn = FNN(
            input_dim=2*meta_node_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
            num_layers=num_layers,
            non_linearity="elu",
            batch_norm=batch_norm
        )

    def get_combination_idxs(self):
        n = self.num_nodes
        idx_grid = torch.stack(
            torch.meshgrid(
                torch.arange(n),
                torch.arange(n)),
            dim=2).view(-1, 2)
        idxs2 = [i * n + j for j in range(n) for i in range(n) if i > j]
        idxs1 = [j * n + i for j in range(n) for i in range(n) if (j * n + i not in idxs2) & (i != j)]
        idxs = torch.stack((idx_grid[idxs1], idx_grid[idxs2]), dim=2)
        return idxs

    def forward(self, x):
        # x: [batch_size, num_meta_nodes, meta_node_dim]
        idxs = self.idxs.type_as(x)
        x = x[torch.arange(x.shape[0]).view(-1, 1, 1, 1), idxs].flatten(start_dim=3).flatten(end_dim=2)
        x = self.fnn(x)
        x = torch.mean(x.view(-1, 2, self.output_dim), dim=1)  # average over both directions
        #x = torch.nn.functional.sigmoid(x)  # probability that two nodes are connected
        return x


class NodePredictor(torch.nn.Module):
    def __init__(self, num_nodes, meta_node_dim, hidden_dim, num_layers, batch_norm=False, num_node_features=24):
        super().__init__()
        self.num_nodes = num_nodes,
        self.meta_node_dim = meta_node_dim
        self.fnn = FNN(
            input_dim=meta_node_dim,
            hidden_dim=hidden_dim,
            output_dim=num_node_features + 1,  # +1 for probability that node exists
            num_layers=num_layers,
            non_linearity="elu",
            batch_norm=batch_norm
        )

    def forward(self, x):
        # x: [batch_size, num_meta_nodes, meta_node_dim]
        x = x.view(-1)
        x = self.fnn(x)
        return x
