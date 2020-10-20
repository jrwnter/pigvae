import torch
from graphae.fully_connected import FNN


class SinkhornNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_nodes, batch_norm=False, non_linearity="elu"):
        super().__init__()
        self.fnn = FNN(
            input_dim=num_nodes * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * num_nodes,
            num_layers=num_layers,
            non_linearity=non_linearity,
            batch_norm=batch_norm
        )

    def forward(self, x):
        # x: node_embeddings [batch_size, num_nodes, node_dim]
        shape = x.shape
        x = x.reshape(shape[0], shape[1] * shape[2])  # TODO: why does view not work here?
        x = self.fnn(x)
        x = x.view(shape[0], shape[1], shape[1])  # [batch_size, num_nodes, num_nodes]
        x = torch.sigmoid(x)
        return x


class Permuter(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_nodes, batch_norm=False, non_linearity="elu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.fnn = FNN(
            input_dim=2 * num_nodes * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * num_nodes,
            num_layers=num_layers,
            non_linearity=non_linearity,
            batch_norm=batch_norm
        )

    def forward(self, node_embs_in, node_embs_out):
        # node_embs_in [batch_size, num_nodes, node_dim]
        x = torch.cat((node_embs_in, node_embs_out), dim=-1).flatten(start_dim=1)
        x = self.fnn(x)
        x = x.view(-1, self.num_nodes, self.num_nodes)
        x = torch.sigmoid(x)
        return x
