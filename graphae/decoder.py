import torch
from graphae.fully_connected import FNN


class EdgeDecoder(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, num_nodes, num_layers,
                 num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.fnn = FNN(
            input_dim=num_nodes * node_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * num_nodes * self.num_edge_features,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, nodes):
        # nodes [batch_size, num_nodes, node_dim]
        batch_size = nodes.size(0)
        x = nodes.flatten(start_dim=1)
        x = self.fnn(x)
        x = x.view(batch_size, self.num_nodes, self.num_nodes, self.num_edge_features)
        x = (x + x.permute(0, 2, 1, 3)) / 2
        return x


class NodePredictor(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, batch_norm=False, num_node_features=24, non_lin="lrelu"):
        super().__init__()
        self.num_node_features = num_node_features + 1  # +1 for probability that node does not exists
        self.fnn = FNN(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_node_features,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
            flatten_for_batch_norm=True
        )

    def forward(self, x):
        # x: [batch_size, num_nodes, input_dim]
        x = self.fnn(x)
        return x
