import numpy as np
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
            input_size=node_dim,
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
            batch_norm=False,
        )
        self.linear_out = Linear(
            in_features=hidden_dim,
            out_features=node_dim
        )

    def forward(self, emb, node_emb_encoded, teacher_forcing):
        # emb [batch_size, emb_dim]
        # node_emb_encoded [batch_size, num_nodes, node_dim]
        node_emb_encoded_ = torch.clone(node_emb_encoded)
        inf = torch.from_numpy(np.array(np.float("inf"))).type_as(node_emb_encoded_)
        batch_size = emb.size(0)
        hx = self.fnn_in(emb)
        hx = hx.view(-1, self.num_layers_rnn, self.hidden_dim).permute(1, 0, 2).contiguous()
        cx = torch.zeros_like(hx)
        decoder_input = torch.zeros((batch_size, self.node_dim), device=emb.device)  # TODO: Change start "token"?
        node_emb_pred = []
        idxs = []
        for i in range(self.num_nodes):
            x, (hx, cx) = self.rnn(decoder_input.unsqueeze(0), (hx, cx))
            #x = self.fnn_out(x[0])
            x = self.linear_out(x[0])
            node_emb_pred.append(x)
            idx = torch.argmin(torch.norm(x.unsqueeze(1) - node_emb_encoded_, dim=-1), dim=-1)  # [batch_size]
            idxs.append(idx)
            x_ = node_emb_encoded[torch.arange(batch_size), idx]
            node_emb_encoded_[torch.arange(batch_size), idx] = inf
            if torch.rand(1) < teacher_forcing:
                decoder_input = x_
            else:
                decoder_input = x
        node_emb_pred = torch.stack(node_emb_pred, dim=1)
        idxs = torch.stack(idxs, dim=1)
        perm = torch.zeros(batch_size, self.num_nodes, self.num_nodes).type_as(idxs).scatter_(2, idxs.unsqueeze(-1), 1)
        # transpose perm since we want to aply it on the predicted nodes
        perm = perm.transpose(1, 2).float()
        node_emb_pred = torch.matmul(perm, node_emb_pred)
        return node_emb_pred


class EdgeDecoder(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, num_nodes, num_layers,
                 num_edge_features, non_lin, batch_norm):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_edge_features = num_edge_features + 1
        self.fnn = FNN(
            input_dim=num_nodes * node_dim,
            hidden_dim=hidden_dim,
            output_dim=num_nodes * num_nodes * self.num_edge_features,
            num_layers=num_layers,
            non_linearity=non_lin,
            batch_norm=batch_norm,
        )

    def forward(self, nodes):
        # nodes [batch_size, num_nodes, node_dim]
        batch_size = nodes.size(0)
        x = nodes.flatten(start_dim=1)
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
            flatten_for_batch_norm=True
        )

    def forward(self, x):
        # x: [batch_size, num_nodes, input_dim]
        x = self.fnn(x)
        return x
