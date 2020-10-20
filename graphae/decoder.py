import torch
from graphae.fully_connected import FNN
from torch.nn import LSTM, GRU, GRUCell, Linear


# TODO: More RNN layers?
class NodeEmbDecoder(torch.nn.Module):
    def __init__(self, emb_dim, node_dim, hidden_dim, num_nodes, num_layers_fnn, num_layers_rnn,
                 non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers_rnn = num_layers_rnn
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.fnn_in = FNN(
            input_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=num_layers_rnn * hidden_dim,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )
        self.rnn = LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_rnn,
            batch_first=False,
        )
        self.fnn_out = self.fnn = FNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=node_dim,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, emb):
        hx = self.fnn_in(emb)
        hx = hx.view(-1, self.num_layers_rnn, self.hidden_dim).permute(1, 0, 2).contiguous()
        cx = torch.zeros_like(hx)
        out = [torch.zeros((1, hx.size(1), hx.size(2)), device=emb.device)]
        for i in range(self.num_nodes):
            x, (hx, cx) = self.rnn(out[-1], (hx, cx))
            out.append(x)
        x = torch.cat(out[1:], dim=0)
        x = x.permute(1, 0, 2).flatten(end_dim=-2)
        x = self.fnn_out(x)
        x = x.view(-1, self.num_nodes, self.node_dim)
        return x


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


"""class EdgeDecoder(torch.nn.Module):
    def __init__(self, emb_dim, node_dim, hidden_dim, num_nodes, num_layers_rnn, num_layers_fnn,
                 num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_layers_rnn = num_layers_rnn
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.fnn_init_state = FNN(
            input_dim=emb_dim + num_nodes * node_dim,
            hidden_dim=hidden_dim,
            output_dim=num_layers_rnn * hidden_dim,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )
        self.fnn_couple_nodes = FNN(
            input_dim=2*node_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )
        self.rnn = LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_rnn,
            batch_first=True,
            bidirectional=True
        )

        self.fnn_out = FNN(
            input_dim=2*hidden_dim,
            hidden_dim=2*hidden_dim,
            output_dim=num_nodes * self.num_edge_features,
            num_layers=num_layers_fnn,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )
        self.triu_idxs = torch.triu_indices(num_nodes, num_nodes, offset=0)

    def forward(self, emb, nodes):
        # emb [batch_size, emb_dim]
        # nodes [batch_size, num_nodes, input_dim/node_dim]
        batch_size = nodes.size(0)
        x = torch.cat((emb, nodes.flatten(start_dim=1)), dim=-1)
        hx = self.fnn_init_state(x)
        hx = hx.view(-1, self.num_layers_rnn, self.hidden_dim).permute(1, 0, 2).contiguous()
        cx = torch.zeros_like(hx)
        x = torch.cat(
            (nodes[torch.arange(batch_size).view(-1, 1), self.triu_idxs[0].view(1, -1)],
             nodes[torch.arange(batch_size).view(-1, 1), self.triu_idxs[1].view(1, -1)]),
            dim=-1
        )
        x, _ = self.rnn(x, (hx, cx))
        x = self.fnn_out(x)
        out = torch.zeros((batch_size, self.num_nodes, self.num_nodes, self.num_edge_features), device=x.device)
        out[torch.arange(batch_size).view(-1, 1), self.triu_idxs[0].view(1, -1), self.triu_idxs[1].view(1, -1)] = x
        out[torch.arange(batch_size).view(-1, 1), self.triu_idxs[1].view(1, -1), self.triu_idxs[0].view(1, -1)] = x
        return out"""


class EdgeDecoder(torch.nn.Module):
    def __init__(self, emb_dim, node_dim, hidden_dim, num_nodes, num_layers,
                 num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.fnn = FNN(
            input_dim=emb_dim + num_nodes * node_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * num_nodes * self.num_edge_features,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, emb, nodes):
        # emb [batch_size, emb_dim]
        # nodes [batch_size, num_nodes, input_dim/node_dim]
        batch_size = nodes.size(0)
        x = torch.cat((emb, nodes.flatten(start_dim=1)), dim=-1)
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
            flatten_for_batch_norm=False
        )

    def forward(self, x):
        # x: [batch_size, num_nodes, input_dim]
        num_nodes = x.size(1)
        x = x.flatten(end_dim=-2)
        x = self.fnn(x)
        x = x.view(-1, num_nodes, self.num_node_features)
        return x
