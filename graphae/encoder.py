import torch
from torch.nn import Linear, ELU, Sequential
from torch_geometric.nn import GINConv, NNConv
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
    def __init__(self, input_dim, hidden_dim_gnn, num_layers_gnn, hidden_dim_fnn, num_layers_fnn, node_dim,
                 stack_node_emb, num_edge_features, num_nodes, batch_norm=False, non_linearity="elu"):
        super().__init__()
        self.node_dim = node_dim
        self.stack_node_emb = stack_node_emb
        self.num_layers_gnn = num_layers_gnn
        self.num_nodes = num_nodes
        self.batch_norm = batch_norm
        conv_layers = [EdgeNNConv(input_dim, hidden_dim_gnn, num_edge_features)]
        conv_layers += [GraphConv(
            input_dim=hidden_dim_gnn,
            hidden_dim=hidden_dim_gnn,
            output_dim=hidden_dim_gnn,
            num_mlp_layers=3,
            batch_norm=batch_norm,
            non_linearity=non_linearity) for _ in range(num_layers_gnn - 1)
        ]
        self.conv_layers = torch.nn.ModuleList(conv_layers)

        self.fnn = FNN(
            input_dim=num_layers_gnn * hidden_dim_gnn if self.stack_node_emb else hidden_dim_gnn,
            output_dim=node_dim,
            hidden_dim=hidden_dim_fnn,
            num_layers=num_layers_fnn,
            non_linearity=non_linearity,
            batch_norm=batch_norm,
        )

        if non_linearity == "relu":
            self.non_linearity = torch.nn.ReLU()
        elif non_linearity == "elu":
            self.non_linearity = torch.nn.ELU()
        elif non_linearity == "lrelu":
            self.non_linearity = torch.nn.LeakyReLU()

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        node_emb = []
        for i in range(self.num_layers_gnn):
            if i > 0:
                x = self.non_linearity(x)
            x = self.conv_layers[i](x, edge_index, edge_attr)
            node_emb.append(x)
        node_emb = torch.stack(node_emb, dim=2)

        if self.stack_node_emb:
            node_emb = node_emb.flatten(start_dim=1)
        else:
            node_emb = node_emb[:, :, -1]  # just take output of the last layer otherwise of all layers (see. GIN paper)

        node_emb = self.fnn(node_emb)

        return node_emb
