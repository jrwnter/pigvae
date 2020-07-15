import torch
from torch.nn import Linear, Sequential, ELU, BatchNorm1d, ReLU


class FNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, non_linearity="relu", batch_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        layers = [Linear(input_dim, hidden_dim)]
        layers += [Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        layers += [Linear(hidden_dim, output_dim)]
        self.layers = torch.nn.ModuleList(layers)
        if batch_norm:
            bn_layers = [BatchNorm1d(input_dim)]
            bn_layers += [BatchNorm1d(hidden_dim) for _ in range(num_layers - 2)]
            self.bn_layers = torch.nn.ModuleList(bn_layers)
        if non_linearity == "relu":
            self.non_linearity = torch.nn.ReLU
        elif non_linearity == "elu":
            self.non_linearity = torch.nn.ELU()

    def forward(self, x):
        x = self.non_linearity(x)
        for i in range(self.num_layers):
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.non_linearity(x)
            x = self.layers[i](x)
        return x


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
    def __init__(self, num_nodes, meta_node_dim, hidden_dim, num_layers, batch_norm=False, num_edge_features=5):
        super().__init__()
        self.num_nodes = num_nodes,
        self.meta_node_dim = meta_node_dim
        self.output_dim = num_edge_features + 1  # +1 for probability that it exists
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
    def __init__(self, num_nodes, meta_node_dim, hidden_dim, num_layers, batch_norm=False, num_node_features=21):
        super().__init__()
        self.num_nodes = num_nodes,
        self.meta_node_dim = meta_node_dim
        self.fnn = FNN(
            input_dim=meta_node_dim,
            hidden_dim=hidden_dim,
            output_dim=num_node_features + 1,  # +1 for probability that it exists
            num_layers=num_layers,
            non_linearity="elu",
            batch_norm=batch_norm
        )

    def forward(self, x):
        # x: [batch_size, num_meta_nodes, meta_node_dim]
        x = x.view(-1)
        x = self.fnn(x)
        return x
