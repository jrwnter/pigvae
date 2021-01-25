import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss, MSELoss, TripletMarginLoss
from pytorch_lightning.metrics.classification import Recall, Precision, Accuracy


MEAN_DISTANCE = 2.0626
STD_DISTANCE = 1.1746


# TODO: make metric for loss. Right now does not sync correctly, I guess?
class Critic(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.reconstruction_loss = GraphReconstructionLoss()
        self.perm_loss = PermutaionMatrixPenalty()

    def forward(self, graph_true, graph_pred, perm):
        recon_loss = self.reconstruction_loss(
            graph_true=graph_true,
            graph_pred=graph_pred
        )
        perm_loss = self.perm_loss(perm)
        loss = {**recon_loss, "perm_loss": perm_loss}
        loss["loss"] = loss["loss"] + 0.1 * perm_loss
        return loss

    def evaluate(self, graph_true, graph_pred, perm, prefix=None):
        loss = self(
            graph_true=graph_true,
            graph_pred=graph_pred,
            perm=perm,
        )
        metrics = loss

        if prefix is not None:
            metrics2 = {}
            for key in metrics.keys():
                new_key = prefix + "_" + str(key)
                metrics2[new_key] = metrics[key]
            metrics = metrics2
        return metrics


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_loss = BCEWithLogitsLoss()

    def forward(self, graph_true, graph_pred):
        mask = graph_true.mask
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        edges_true = (graph_true.edge_features[adj_mask] == 1).float()
        edges_pred = graph_pred.edge_features[adj_mask]
        edge_loss = self.edge_loss(
            input=edges_pred,
            target=edges_true
        )
        loss = {
            "edge_loss": edge_loss,
            "loss": edge_loss
        }
        return loss


class PropertyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = MSELoss()

    def forward(self, input, target):
        loss = self.mse_loss(
            input=input,
            target=target
        )
        return loss


class PermutaionMatrixPenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(p, axis, normalize=True, eps=10e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = - torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    def forward(self, perm, eps=10e-8):
        #print(perm.shape)
        perm = perm + eps
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        loss = entropy_col.mean() + entropy_row.mean()
        return loss


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


def node_balanced_accuracy(nodes_pred, nodes_true):
    element_type_true, element_type_pred = nodes_true[:, :11], nodes_pred[:, :11]
    charge_type_true, charge_type_pred = nodes_true[:, 11:16], nodes_pred[:, 11:16]
    hybridization_type_true, hybridization_type_pred = nodes_true[:, 16:], nodes_pred[:, 16:]
    element_type_acc = scipy_balanced_accuracy(element_type_pred, element_type_true)
    charge_type_acc = scipy_balanced_accuracy(charge_type_pred, charge_type_true)
    hybridization_type_acc = scipy_balanced_accuracy(hybridization_type_pred, hybridization_type_true)
    return element_type_acc, charge_type_acc, hybridization_type_acc


def edge_balanced_accuracy(edges_pred, edges_true):
    acc = scipy_balanced_accuracy(edges_pred, edges_true)
    return acc
