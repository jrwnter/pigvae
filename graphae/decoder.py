import torch
from graphae.fully_connected import FNN
from pivae.dds import DDSBlock





class EdgeDecoder(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, num_nodes, num_layers,
                 num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.dds_block = DDSBlock(
            input_dim=2*node_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_edge_features,
            num_layers=num_layers,
            aggregate_inner="max",
            aggregate_outer=None,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, x, edge_index, edge_index_batch):
        # x [batch_size, node_dim]
        # edge_index: dense edge_index (all combinations of num_nodes)
        # edge_index_batch: batch_idx of edge index
        x = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        x = self.dds_block(x, edge_index_batch)
        return x


class NodePredictor(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, batch_norm=False, num_node_features=24, non_lin="lrelu"):
        super().__init__()
        self.num_node_features = num_node_features
        self.fnn = FNN(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_node_features,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fnn(x)
        return x


class SelfAttention()