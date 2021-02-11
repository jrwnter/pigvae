import torch
from torch.nn import CrossEntropyLoss, MSELoss


ELEMENT_TYPE_WEIGHTS = torch.Tensor([2.8469,  17.1120,  12.8389, 720.3815, 1.9568])
CHARGE_TYPE_WEIGHTS = torch.Tensor([310.5101, 122.4133, 1.3623])
#EDGE_WEIGHTS = torch.Tensor([1.0855,  14.0608, 385.1767, 851.0145, 260.9370])
EDGE_WEIGHTS = torch.Tensor([1.,  2, 10, 20, 4])


class Critic(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.alpha = hparams["kld_loss_scale"]
        self.beta = hparams["perm_loss_scale"]
        self.gamma = hparams["property_loss_scale"]
        self.vae = hparams["vae"]
        self.reconstruction_loss = GraphReconstructionLoss()
        self.perm_loss = PermutaionMatrixPenalty()
        self.property_loss = PropertyLoss()
        self.kld_loss = KLDLoss()

    def forward(self, graph_true, graph_pred, perm, mu, logvar):
        recon_loss = self.reconstruction_loss(
            graph_true=graph_true,
            graph_pred=graph_pred
        )
        perm_loss = self.perm_loss(perm)
        property_loss = self.property_loss(
            input=graph_pred.properties,
            target=graph_true.properties
        )
        loss = {**recon_loss, "perm_loss": perm_loss, "property_loss": property_loss}
        loss["loss"] = loss["loss"] + self.beta * perm_loss + self.gamma * property_loss
        if self.vae:
            kld_loss = self.kld_loss(mu, logvar)
            loss["kld_loss"] = kld_loss
            loss["loss"] = loss["loss"] + self.alpha * kld_loss
        return loss

    def evaluate(self, graph_true, graph_pred, perm, mu, logvar, prefix=None):
        loss = self(
            graph_true=graph_true,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar
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
        self.element_type_loss = CrossEntropyLoss(weight=ELEMENT_TYPE_WEIGHTS)
        self.charge_type_loss = CrossEntropyLoss(weight=CHARGE_TYPE_WEIGHTS)
        self.edge_type_loss = CrossEntropyLoss(weight=EDGE_WEIGHTS)
        self.distance_loss = MSELoss()

    def forward(self, graph_true, graph_pred):
        mask = graph_true.mask
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        nodes_true = graph_true.node_features[mask][:, :10]
        nodes_pred = graph_pred.node_features[mask][:, :10]
        edge_types_true = graph_true.edge_features[adj_mask][:, :5]
        edge_types_pred = graph_pred.edge_features[adj_mask][:, :5]
        distances_true = graph_true.edge_features[adj_mask][:, -1]
        distances_pred = graph_pred.edge_features[adj_mask][:, -1]

        element_type_pred = nodes_pred[:, :5]
        element_type_true = torch.argmax(nodes_true[:, :5], axis=-1).flatten()
        element_type_loss = self.element_type_loss(
            input=element_type_pred,
            target=element_type_true
        )
        charge_type_pred = nodes_pred[:, 5:8]
        charge_type_true = torch.argmax(nodes_true[:, 5:8], axis=-1).flatten()
        charge_type_loss = self.charge_type_loss(
            input=charge_type_pred,
            target=charge_type_true
        )
        edge_types_true = torch.argmax(edge_types_true, axis=-1).flatten()
        edge_type_loss = self.edge_type_loss(
            input=edge_types_pred,
            target=edge_types_true
        )
        distance_loss = self.distance_loss(
            input=distances_pred,
            target=distances_true
        )
        node_loss = (element_type_loss + charge_type_loss) / 2
        total_loss = node_loss + edge_type_loss + distance_loss
        loss = {
            "element_type_loss": element_type_loss,
            "charge_type_loss": charge_type_loss,
            "edge_type_loss": edge_type_loss,
            "distance_loss": distance_loss,
            "node_loss": node_loss,
            "loss": total_loss
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


class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss


"""
from sklearn.metrics import balanced_accuracy_score
import numpy as np

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
    element_type_true, element_type_pred = nodes_true[:, :5], nodes_pred[:, :5]
    charge_type_true, charge_type_pred = nodes_true[:, 5:8], nodes_pred[:, 5:8]
    element_type_acc = scipy_balanced_accuracy(element_type_pred, element_type_true)
    charge_type_acc = scipy_balanced_accuracy(charge_type_pred, charge_type_true)
    return element_type_acc, charge_type_acc


def edge_balanced_accuracy(edges_pred, edges_true):
    acc = scipy_balanced_accuracy(edges_pred, edges_true)
    return acc
    
"""
