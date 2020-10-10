import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss


ELEMENT_TYPE_WEIGHTS = torch.Tensor(
    [1.1331e-04, 7.1108e-04, 8.0928e-04, 4.9133e-03, 5.2829e-03, 2.5689e-01,
     2.7862e-01, 8.5392e-03, 1.9235e-02, 1.4627e-01, 2.7862e-01])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2605, 0.2605, 0.2180, 0.2605, 0.0006])
HYBRIDIZATION_TYPE_WEIGHT = torch.Tensor(
    [2.4990e-01, 3.8090e-04, 3.9730e-06, 6.6788e-06, 2.4990e-01, 2.4990e-01, 2.4990e-01])
EDGE_WEIGHTS = torch.Tensor([5.3680e-03, 4.4790e-02, 9.4289e-01, 6.8177e-03, 1.3267e-04])
MASK_POS_WEIGHT = torch.Tensor([0.3221])


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = GraphReconstructionLoss()
        self.permutation_matrix_penalty = PermutaionMatrixPenalty()

    def forward(self, nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred, perm):
        recon_loss = self.reconstruction_loss(nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred)
        perm_loss = self.permutation_matrix_penalty(perm)
        loss = {**recon_loss, "perm_loss": perm_loss}
        loss["loss"] += loss["perm_loss"]
        return loss


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self):
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


class PermutaionMatrixPenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def row_col_penalty(self, perm, axis):
        # v: [batch_size, num_nodes, num_nodes]
        loss = torch.sum(torch.sum(torch.abs(perm), axis=axis) - torch.sqrt(torch.sum(torch.pow(perm, 2), axis=axis)), axis=1)
        return loss

    def forward(self, perm):
        batch_size = perm.size(0)
        num_nodes = perm.size(1)
        identity = torch.ones((batch_size, num_nodes)).type_as(perm)
        penalty = self.row_col_penalty(perm, axis=1) + self.row_col_penalty(perm, axis=2)
        constrain_col = torch.abs(torch.sum(perm, axis=1) - identity).mean(axis=1)
        constrain_row = torch.abs(torch.sum(perm, axis=2) - identity).mean(axis=1)
        #constrain_pos = torch.min(torch.zeros_like(perm), perm).mean(axis=(1, 2))
        constrain = constrain_col + constrain_row #+ constrain_pos
        loss = penalty + constrain
        loss = loss.mean()
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


def node_balanced_accuracy(nodes_pred, nodes_true, mask):
    nodes_true, nodes_pred = nodes_true[mask], nodes_pred[mask]
    element_type_true, element_type_pred = nodes_true[:, :11], nodes_pred[:, :11]
    charge_type_true, charge_type_pred = nodes_true[:, 11:16], nodes_pred[:, 11:16]
    hybridization_type_true, hybridization_type_pred = nodes_true[:, 16:], nodes_pred[:, 16:]
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
    adj_true, adj_pred = adj_true[adj_mask], adj_pred[adj_mask]
    acc = scipy_balanced_accuracy(adj_pred, adj_true)
    return acc


