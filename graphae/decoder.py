import torch
from torch.nn import Linear, BatchNorm1d
from graphae.fully_connected import FNN
from graphae.graph_transformer import TransformerConv
from pivae.dds import DDSBlock


class NodeEmbDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, non_lin, batch_norm):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.layers = [TransformerConv(input_dim, hidden_dim, num_heads)]
        self.layers += [TransformerConv(num_heads * hidden_dim, hidden_dim, num_heads) for _ in range(num_layers - 1)]
        self.layers = torch.nn.ModuleList(self.layers)
        self.linear_out = Linear(num_heads * hidden_dim, output_dim)
        if batch_norm:
            bn_layers = [BatchNorm1d(num_heads * hidden_dim) for _ in range(num_layers)]
            bn_layers += [BatchNorm1d(output_dim)]
            self.bn_layers = torch.nn.ModuleList(bn_layers)
        if non_lin == "relu":
            self.non_lin = torch.nn.ReLU()
        elif non_lin == "elu":
            self.non_lin = torch.nn.ELU()
        elif non_lin == "lrelu":
            self.non_lin = torch.nn.LeakyReLU()

    def forward(self, x, edge_index):
        # x [batch_size, node_dim]
        # edge_index: dense edge_index (all combinations of num_nodes)
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.non_lin(x)
        x = self.linear_out(x)
        x = self.bn_layers[-1](x)
        x = self.non_lin(x)
        return x


class EdgeTypePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, non_lin, batch_norm):
        super().__init__()
        self.output_dim = output_dim
        self.fnn = FNN(
            input_dim=2*input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, x, dense_edge_index):
        # x: nodes from graph decoder [batch_size, node_dim]
        # dense_edge_index: dense edge_index (all combinations of num_nodes)
        x = torch.cat((x[dense_edge_index[0]], x[dense_edge_index[1]]), dim=-1)
        x = self.fnn(x)  # [num_dense_edges, num_edges_types + 1]
        # we defined dense_edge_index in such a way, that both directional follow after each other --> we can average
        # every 2 edges and repeat to get same prediction in both directions
        x = x.view(-1, 2, self.output_dim).mean(dim=1)
        x = torch.repeat_interleave(x, 2, dim=0)
        return x


class NodeTypePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_norm, non_lin):
        super().__init__()
        self.fnn = FNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, x):
        # x: nodes from graph decoder [batch_size, node_dim]
        x = self.fnn(x)
        return x
