import torch
from torch.nn import Linear, BatchNorm1d
from graphae.fully_connected import FNN
from torch_geometric.nn import GATConv
from pivae.dds import DDSBlock

import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (float, optional): If set, will combine aggregation and
            skip information via :math:`\beta\,\mathbf{W}_1 \vec{x}_i + (1 -
            \beta) \left(\sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2
            \vec{x}_j \right)`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 beta: Optional[float] = None, dropout: float = 0.,
                 edge_dim: Optional[int] = None, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = Linear(1, 1)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        x = self.lin_skip(x[1])
        if self.beta is not None:
            out = self.beta * x + (1 - self.beta) * out
        else:
            out += x

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor],
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        query = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        key = self.lin_query(x_i).view(-1, self.heads, self.out_channels)

        lin_edge = self.lin_edge
        if edge_attr is not None:
            edge_attr = lin_edge(edge_attr).view(-1, self.heads,
                                                 self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


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
