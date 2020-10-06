import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss
from graphae.data import add_empty_node_type, add_empty_edge_type


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self, num_edge_types=1, num_node_types=11):
        super().__init__()
        self.node_loss = CrossEntropyLoss()  # TODO: add weights for elements
        self.adj_loss = CrossEntropyLoss()
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types

    def forward(self, nodes_true, adj_true, nodes_pred, adj_pred):
        nodes_true = add_empty_node_type(nodes_true)
        nodes_true = torch.argmax(nodes_true, axis=-1).flatten()
        adj_true = add_empty_edge_type(adj_true)
        adj_true = torch.argmax(adj_true, axis=-1).flatten()

        node_loss = self.node_loss(
            input=nodes_pred.view(-1, self.num_node_types + 1),
            target=nodes_true
        )
        adj_loss = self.adj_loss(
            input=adj_pred.view(-1, self.num_edge_types + 1),
            target=adj_true
        )
        return node_loss, adj_loss


def node_accuracy(input, target):
    target = add_empty_node_type(target)
    acc = (input == target).float().mean()
    return acc


def adj_accuracy(input, target):
    acc = (input == target).float().mean()
    return acc


def node_balanced_accuracy(input, target):
    device = input.device
    target = add_empty_node_type(target)
    target = torch.argmax(target, axis=-1).flatten().cpu().numpy()
    input = torch.argmax(input, axis=-1).flatten().cpu().numpy()
    acc = balanced_accuracy_score(y_true=target, y_pred=input)
    acc = torch.from_numpy(np.array(acc)).to(device).float()
    return acc


def adj_balanced_accuracy(input, target):
    device = input.device
    target = add_empty_edge_type(target)
    target = torch.argmax(target, axis=-1).flatten().cpu().numpy()
    input = torch.argmax(input, axis=-1).flatten().cpu().numpy()
    acc = balanced_accuracy_score(y_true=target, y_pred=input)
    acc = torch.from_numpy(np.array(acc)).to(device).float()
    return acc


