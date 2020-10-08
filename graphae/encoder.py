import torch
from torch.nn import Linear, ELU, Sequential
from torch_geometric.nn import GINConv, NNConv, global_add_pool
from torch_scatter import scatter_add
from graphae.fully_connected import FNN


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


class GraphConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_mlp_layers,
                 batch_norm, non_linearity="relu", dropout=None):
        super().__init__()
        mlp = FNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            non_linearity=non_linearity,
            batch_norm=batch_norm,
            flatten_for_batch_norm=True,
            dropout=dropout
        )
        self.gnn = GINConv(
            nn=mlp,
            eps=0.0,
            train_eps=False,
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.gnn(x, edge_index)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, node_dim, stack_node_emb, num_edge_features,
                 num_layers, num_nodes, batch_norm=False, non_linearity="elu"):
        super().__init__()
        self.node_dim = node_dim
        self.stack_node_emb = stack_node_emb
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.batch_norm = batch_norm
        conv_layers = [EdgeNNConv(input_dim, node_dim, num_edge_features)]
        conv_layers += [GraphConv(node_dim, node_dim, hidden_dim, 3, batch_norm, non_linearity) for _ in range(num_layers - 2)]
        conv_layers += [GraphConv(node_dim, node_dim, hidden_dim, 3, batch_norm, non_linearity)]
        self.conv_layers = torch.nn.ModuleList(conv_layers)

        if non_linearity == "relu":
            self.non_linearity = torch.nn.ReLU()
        elif non_linearity == "elu":
            self.non_linearity = torch.nn.ELU()
        elif non_linearity == "lrelu":
            self.non_linearity = torch.nn.LeakyReLU()

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        node_emb = []
        for i in range(self.num_layers):
            if i > 0:
                x = self.non_linearity(x)
            x = self.conv_layers[i](x, edge_index, edge_attr)
            node_emb.append(x)
        node_emb = torch.stack(node_emb, dim=2)
        if not self.stack_node_emb:
            node_emb = node_emb[:, :, -1]  # just take output of the last layer otherwise of all layers (see. GIN paper)
        return node_emb


class NodeAggregator(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, batch_norm=False, non_linearity="elu"):
        super().__init__()
        self.fnn = FNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            num_layers=num_layers,
            non_linearity=non_linearity,
            batch_norm=batch_norm
        )

    def forward(self, x, batch_idxs):
        # Aggreaate node embeddings for each node to one embedding for whole graph. Flatten takes care of
        # potential stacked node embs.
        x = global_add_pool(x, batch_idxs).flatten(start_dim=1)
        x = self.fnn(x)
        return x
