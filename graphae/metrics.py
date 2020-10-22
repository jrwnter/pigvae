import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss, MSELoss, TripletMarginLoss


# 32
"""ELEMENT_TYPE_WEIGHTS = torch.Tensor([
    1.1111e-04, 6.9739e-04, 7.9399e-04, 4.8409e-03, 5.1567e-03, 2.6155e-01,
    2.7761e-01, 8.4134e-03, 1.8924e-02, 1.4429e-01, 2.7761e-01])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2596, 0.2596, 0.2205, 0.2596, 0.0006])
HYBRIDIZATION_TYPE_WEIGHT = torch.Tensor(
    [2.4989e-01, 4.4300e-04, 4.6720e-06, 7.8765e-06, 2.4989e-01, 2.4989e-01, 2.4989e-01])
EDGE_WEIGHTS = torch.Tensor([5.4339e-03, 4.5212e-02, 9.4218e-01, 6.8846e-03, 2.8955e-04])
MASK_POS_WEIGHT = torch.Tensor([0.3221])

# 16"""

ELEMENT_TYPE_WEIGHTS = torch.Tensor(
    [0.0003, 0.0017, 0.0021, 0.0103, 0.0120, 0.2837, 0.2500, 0.0170, 0.0247,
     0.1144, 0.2837])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2594, 0.2594, 0.2211, 0.2594, 0.0006])
HYBRIDIZATION_TYPE_WEIGHT = torch.Tensor(
    [3.3249e-01, 5.4122e-06, 1.4856e-07, 1.5068e-07, 2.5381e-03, 3.3249e-01, 3.3249e-01])
EDGE_WEIGHTS = torch.Tensor([0.0091, 0.0891, 0.8833, 0.0175, 0.0009])
MASK_POS_WEIGHT = torch.Tensor([0.1214])


class Critic(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.reconstruction_loss = GraphReconstructionLoss()
        self.node_emb_matching_loss = NodeEmbMatchingLoss()
        self.alpha = alpha

    def forward(self, nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred, node_emb_enc, node_emb_dec):
        recon_loss = self.reconstruction_loss(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred
        )
        node_emb_matching_loss = self.node_emb_matching_loss(
            node_emb_enc=node_emb_enc,
            node_emb_dec=node_emb_dec
        )
        loss = {**recon_loss, "node_emb_matching_loss": node_emb_matching_loss}
        loss["loss"] += self.alpha * loss["node_emb_matching_loss"]

        return loss

    def evaluate(self, nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred,
                 node_emb_enc, node_emb_dec, prefix=None):
        loss = self(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
            node_emb_enc=node_emb_enc,
            node_emb_dec=node_emb_dec,
        )
        element_type_acc, charge_type_acc, hybridization_type_acc = node_balanced_accuracy(
            nodes_pred=nodes_pred,
            nodes_true=nodes_true,
            mask=mask_true
        )
        adj_acc = adj_balanced_accuracy(
            adj_pred=adj_pred,
            adj_true=adj_true,
            mask=mask_true
        )
        mask_acc = mask_balenced_accuracy(
            mask_pred=mask_pred,
            mask_true=mask_true
        )
        output = {
            **loss,
            "element_type_acc": element_type_acc,
            "charge_type_acc": charge_type_acc,
            "hybridization_type_acc": hybridization_type_acc,
            "adj_acc": adj_acc,
            "mask_acc": mask_acc,
        }
        if prefix is not None:
            output2 = {}
            for key in output.keys():
                new_key = prefix + "_" + str(key)
                output2[new_key] = output[key]
            output = output2
        return output


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


class NodeEmbMatchingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TripletMarginLoss()

    def forward(self, node_emb_enc, node_emb_dec, eps=10e-10):
        batch_size = node_emb_enc.size(0)
        num_nodes = node_emb_enc.size(1)
        dm = torch.norm(node_emb_enc.unsqueeze(1) - node_emb_dec.unsqueeze(2), dim=-1)
        dia = torch.diagonal(dm, 0, dim1=1, dim2=2)  # [batch_size, num_nodes]
        off_dia_mean = dm.masked_fill(torch.eye(num_nodes).view(1, num_nodes, num_nodes).repeat(batch_size, 1, 1).type_as(dm).bool(), 0)
        off_dia_mean = off_dia_mean.mean(dim=-1)
        #print(dia_sum.shape, off_dia_mean.shape)
        loss = dia / (off_dia_mean + eps)
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
    mask_pred = mask_pred.flatten() > 0
    mask_pred = mask_pred.long()
    acc = scipy_balanced_accuracy(mask_pred, mask_true)
    return acc


def adj_balanced_accuracy(adj_pred, adj_true, mask):
    adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    adj_true, adj_pred = adj_true[adj_mask], adj_pred[adj_mask]
    acc = scipy_balanced_accuracy(adj_pred, adj_true)
    return acc
