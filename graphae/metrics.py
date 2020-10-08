import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from graphae.data import add_empty_node_type, add_empty_edge_type


ELEMENT_TYPE_WEIGHTS = torch.Tensor(
    [0.0003, 0.0017, 0.0021, 0.0103, 0.0120, 0.2838, 0.2498, 0.0169, 0.0247, 0.1133, 0.2838])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2603, 0.2603, 0.2185, 0.2603, 0.0006])
HYBRIDIZATION_TYPE_WEIGHT = torch.Tensor(
    [2.4987e-01, 4.8807e-04, 5.0471e-06, 8.5090e-06, 2.4987e-01, 2.4987e-01,
        2.4987e-01])

EDGE_WEIGHTS = torch.Tensor([5.2475e-03, 4.3232e-02, 9.4001e-01, 6.6112e-03, 4.7667e-03, 1.2940e-04])
MASK_POS_WEIGHT = torch.Tensor([0.3216])


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self, num_edge_types=1, num_node_types=11):
        super().__init__()
        self.mask_loss = BCEWithLogitsLoss(pos_weight=MASK_POS_WEIGHT)
        self.element_type_loss = CrossEntropyLoss(weight=ELEMENT_TYPE_WEIGHTS)
        self.charge_type_loss = CrossEntropyLoss(weight=CHARGE_TYPE_WEIGHTS)
        self.hybridization_type_loss = CrossEntropyLoss(weight=HYBRIDIZATION_TYPE_WEIGHT)
        self.adj_loss = CrossEntropyLoss(weight=EDGE_WEIGHTS)

    def forward(self, nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred):
        mask_loss = self.mask_loss(
            input=mask_pred.flatten(),
            target=mask_true.flatten().float()
        )
        element_type_pred = nodes_pred[:, :, :11][mask_true]
        element_type_true = torch.argmax(nodes_true[:, :, :11][mask_true], axis=-1).flatten()
        element_type_loss = self.element_type_loss(
            input=element_type_pred,
            target=element_type_true
        )
        charge_type_pred = nodes_pred[:, :, 11:16][mask_true]
        charge_type_true = torch.argmax(nodes_true[:, :, 11:16][mask_true], axis=-1).flatten()
        charge_type_loss = self.charge_type_loss(
            input=charge_type_pred,
            target=charge_type_true
        )
        hybridization_type_pred = nodes_pred[:, :, 16:][mask_true]
        hybridization_type_true = torch.argmax(nodes_true[:, :, 16:][mask_true], axis=-1).flatten()
        hybridization_type_loss = self.hybridization_type_loss(
            input=hybridization_type_pred,
            target=hybridization_type_true
        )

        adj_mask = mask_true.unsqueeze(1) * mask_true.unsqueeze(2)
        adj_pred = adj_pred[adj_mask]
        adj_true = add_empty_edge_type(adj_true)
        adj_true = torch.argmax(adj_true[adj_mask], axis=-1).flatten()
        adjacency_loss = self.adj_loss(
            input=adj_pred,
            target=adj_true
        )
        node_loss = (element_type_loss + charge_type_loss + hybridization_type_loss) / 3
        total_loss = mask_loss + node_loss + adjacency_loss
        loss = {
            "mask_loss": mask_loss,
            "element_type_loss": element_type_loss,
            "charge_type_loss": charge_type_loss,
            "hybridization_type_loss": hybridization_type_loss,
            "adjacency_loss": adjacency_loss,
            "node_loss": node_loss,
            "loss": total_loss
        }
        return loss


def node_accuracy(input, target):
    target = add_empty_node_type(target)
    acc = (input == target).float().mean()
    return acc


def adj_accuracy(input, target):
    acc = (input == target).float().mean()
    return acc


def scipy_balanced_accuracy(input, target):
    device = input.device
    if input.dim() == 2:
        target = torch.argmax(target, axis=-1)
        input = torch.argmax(input, axis=-1)
    elif input.dim() > 2:
        print(input.shape)
        raise ValueError
    target = target.cpu().numpy()
    input = input.cpu().numpy()
    acc = balanced_accuracy_score(y_true=target, y_pred=input)
    acc = torch.from_numpy(np.array(acc)).to(device).float()
    return acc


def node_balanced_accuracy(nodes_pred, nodes_true, mask):
    nodes_true, nodes_pred = nodes_true[mask], nodes_pred[mask]
    element_type_true, element_type_pred = nodes_true[:, :11], nodes_pred[:, :11]
    charge_type_true, charge_type_pred = nodes_true[:, 11:17], nodes_pred[:, 11:17]
    hybridization_type_true, hybridization_type_pred = nodes_true[:, 17:], nodes_pred[:, 17:]
    element_type_acc = scipy_balanced_accuracy(element_type_pred, element_type_true)
    charge_type_acc = scipy_balanced_accuracy(charge_type_pred, charge_type_true)
    hybridization_type_acc = scipy_balanced_accuracy(hybridization_type_pred, hybridization_type_true)
    return element_type_acc, charge_type_acc, hybridization_type_acc


def mask_balenced_accuracy(mask_pred, mask_true):
    mask_true = mask_true.float().flatten()
    mask_pred = mask_pred.flatten() > 0.5
    mask_pred = mask_pred.long()
    acc = scipy_balanced_accuracy(mask_pred, mask_true)
    return acc


def adj_balanced_accuracy(adj_pred, adj_true, mask):
    adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    adj_true = add_empty_edge_type(adj_true)
    adj_true, adj_pred = adj_true[adj_mask], adj_pred[adj_mask]
    acc = scipy_balanced_accuracy(adj_pred, adj_true)
    return acc


