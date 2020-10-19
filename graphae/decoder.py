import torch
from graphae.fully_connected import FNN
from torch.nn import GRU, GRUCell, Linear


# TODO: More RNN layers?
class MetaNodeRNN(torch.nn.Module):
    def __init__(self, emb_dim, meta_node_dim, hidden_dim, num_nodes, num_layers, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.fnn = FNN(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )
        self.rnn = GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim
        )
        self.out_linear = Linear(
            in_features=hidden_dim,
            out_features=meta_node_dim
        )

    def forward(self, emb):
        hx = self.fnn(emb)
        out = [torch.zeros_like(hx)]
        for i in range(self.num_nodes):
            hx = self.rnn(input=out[-1], hx=hx)
            out.append(hx)
        x = torch.stack(out[1:], dim=1)
        x = self.out_linear(x)
        return x, hx


# TODO: implemet some kind of attention(each RNN step takes all node emb in again?) nested RNN?
"""class EdgeRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers_rnn, num_layers_fnn, h_0_dim, num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers_rnn = num_layers_rnn
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.rnn = GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_rnn,
            batch_first=True,
            bidirectional=True
        )
        self.fnn_in = Linear(
            in_features=h_0_dim,
            out_features=num_layers_rnn * 2 * hidden_dim
        )

        self.fnn_out = FNN(
            input_dim=2*hidden_dim,
            hidden_dim=2*hidden_dim,
            output_dim=num_nodes * self.num_edge_features,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, x, hx):
        # x [batch_size, num_nodes, input_dim/node_dim]
        h_0 = self.fnn_in(hx)
        h_0 = h_0.view(-1, self.num_layers_rnn * 2, self.hidden_dim).permute(1, 0, 2).contiguous()
        x, _ = self.rnn(x, h_0)
        x = self.fnn_out(x)
        x = x.transpose(1, 0)  # back to batch first [batch_size, num_nodes, num_nodes * num_edge_features]
        x = x.reshape(-1, self.num_nodes, self.num_nodes, self.num_edge_features)
        x = (x + x.permute(0, 2, 1, 3)) / 2
        return x"""


class EdgeRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers_rnn, num_layers_fnn, h_0_dim, num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers_rnn = num_layers_rnn
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.rnn1 = GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_rnn,
            batch_first=True,
            bidirectional=True
        )
        self.rnn1 = GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_rnn,
            batch_first=True,
            bidirectional=True
        )
        self.fnn_in = Linear(
            in_features=h_0_dim,
            out_features=num_layers_rnn * 2 * hidden_dim
        )

        self.fnn_out = FNN(
            input_dim=2*hidden_dim,
            hidden_dim=2*hidden_dim,
            output_dim=num_nodes * self.num_edge_features,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, x, hx):
        # x [batch_size, num_nodes, input_dim/node_dim]
        h_0 = self.fnn_in(hx)
        h_0 = h_0.view(-1, self.num_layers_rnn * 2, self.hidden_dim).permute(1, 0, 2).contiguous()
        x, _ = self.rnn(x, h_0)
        x = self.fnn_out(x)
        x = x.transpose(1, 0)  # back to batch first [batch_size, num_nodes, num_nodes * num_edge_features]
        x = x.reshape(-1, self.num_nodes, self.num_nodes, self.num_edge_features)
        x = (x + x.permute(0, 2, 1, 3)) / 2
        return x

class NodePredictor(torch.nn.Module):
    def __init__(self, meta_node_dim, hidden_dim, num_layers, batch_norm=False, num_node_features=24, non_lin="lrelu"):
        super().__init__()
        self.num_node_features = num_node_features + 1  # +1 for probability that node does not exists
        self.fnn = FNN(
            input_dim=meta_node_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_node_features,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
            flatten_for_batch_norm=False
        )

    def forward(self, x):
        # x: [batch_size, num_nodes, input_dim]
        x = self.fnn(x)
        return x
