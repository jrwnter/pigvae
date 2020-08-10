import torch
from graphae.fully_connected import FNN


class EdgePredictor(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers, batch_norm=False, non_lin="lrelu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.fnn = FNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * num_nodes * 2,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.shape[0]
        x = self.fnn(x)
        x = x.view(batch_size, self.num_nodes, self.num_nodes, 2)
        x = (x + x.permute(0, 2, 1, 3)) / 2
        return x


class NodePredictor(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers, batch_norm=False, num_node_features=24, non_lin="lrelu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.output_dim = num_nodes * num_node_features  # +1 for probability that node exists (mask)
        self.fnn = FNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
            flatten_for_batch_norm=False
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fnn(x)
        x = x.view(-1, self.num_nodes, self.num_node_features)
        return x
