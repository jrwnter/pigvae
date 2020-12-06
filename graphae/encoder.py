import torch
from torch.nn import Linear, LayerNorm, Dropout
from graphae.graph_transformer import GraphTransformer
from graphae.fully_connected import FNN


class GraphEncoder(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, node_hidden_dim, edge_hidden_dim,
                 nk_dim, ek_dim, v_dim, num_layers, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([
            GraphSelfAttention(num_heads, node_hidden_dim, edge_hidden_dim, nk_dim, ek_dim, v_dim)
            for _ in range(num_layers)])
        self.node_fc_in = Linear(node_dim, node_hidden_dim)
        self.edge_fc_in = Linear(edge_dim, edge_hidden_dim)
        self.layer_norm_nodes = LayerNorm(node_dim)
        self.layer_norm_edges = LayerNorm(edge_dim)
        self.dropout = Dropout(0.1)

    def forward(self, node_features, edge_features, mask):
        # node_features [batch_size, num_nodes, node_dim]
        # edge_features: [batch_size, num_nodes, num_nodes, node_dim]
        # mask: [batch_size, num_nodes]
        node_features = self.layer_norm_nodes(self.dropout(self.node_fc_in(node_features)))
        edge_features = self.layer_norm_edges(self.dropout(self.edge_fc_in(edge_features)))
        for layer in self.layers:
            node_features, edge_features, _ = layer(node_features, edge_features, mask)
        return node_features, edge_features
