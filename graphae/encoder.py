import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import DenseGINConv
from graphae.fully_connected import FNN


class GraphConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_mlp_layers, batch_norm, non_linearity="relu"):
        super().__init__()
        mlp = FNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            non_linearity=non_linearity,
            batch_norm=batch_norm,
            flatten_for_batch_norm=True
        )
        self.gnn = DenseGINConv(
            nn=mlp,
            eps=0.0,
            train_eps=False
        )

    def forward(self, x, adj, mask):
        x = self.gnn(x, adj, mask)
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, node_dim, emb_dim, num_layers, batch_norm=False, non_linearity="relu"):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        conv_layers = [GraphConv(input_dim, node_dim, hidden_dim, 3, batch_norm, non_linearity)]
        conv_layers += [GraphConv(node_dim, node_dim, hidden_dim, 3, batch_norm, non_linearity) for _ in range(num_layers - 2)]
        conv_layers += [GraphConv(node_dim, node_dim, hidden_dim, 3, batch_norm, non_linearity)]
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.fnn = FNN(
            input_dim=num_layers * node_dim,
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            num_layers=3,
            non_linearity="lrelu",
            batch_norm=batch_norm
        )

        if non_linearity == "relu":
            self.non_linearity = torch.nn.ReLU()
        elif non_linearity == "elu":
            self.non_linearity = torch.nn.ELU()
        elif non_linearity == "lrelu":
            self.non_linearity = torch.nn.LeakyReLU()

    def forward(self, x, adj, mask=None):
        node_emb = []
        for i in range(self.num_layers):
            if i > 0:
                x = self.non_linearity(x)
            x = self.conv_layers[i](x, adj, mask)
            node_emb.append(x)
        stacked_node_emb = torch.stack(node_emb, dim=2)
        graph_emb = torch.sum(stacked_node_emb, dim=1).flatten(start_dim=1)
        graph_emb = self.fnn(graph_emb)
        return graph_emb


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
