import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss, MSELoss, TripletMarginLoss
from pytorch_lightning.metrics.classification import Recall, Precision, Accuracy

# 32
ELEMENT_TYPE_WEIGHTS = torch.Tensor([
    1.1111e-04, 6.9739e-04, 7.9399e-04, 4.8409e-03, 5.1567e-03, 2.6155e-01,
    2.7761e-01, 8.4134e-03, 1.8924e-02, 1.4429e-01, 2.7761e-01])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2596, 0.2596, 0.2205, 0.2596, 0.0006])
HYBRIDIZATION_TYPE_WEIGHT = torch.Tensor(
    [2.4989e-01, 4.4300e-04, 4.6720e-06, 7.8765e-06, 2.4989e-01, 2.4989e-01, 2.4989e-01])
#EDGE_WEIGHTS = torch.Tensor([5.4339e-03, 4.5212e-02, 9.4218e-01, 6.8846e-03, 2.8955e-04])
EDGE_WEIGHTS = torch.Tensor([1, 2, 10, 100, 2])

# 16

"""ELEMENT_TYPE_WEIGHTS = torch.Tensor(
    [0.0003, 0.0017, 0.0021, 0.0103, 0.0120, 0.2837, 0.2500, 0.0170, 0.0247,
     0.1144, 0.2837])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2594, 0.2594, 0.2211, 0.2594, 0.0006])
HYBRIDIZATION_TYPE_WEIGHT = torch.Tensor(
    [3.3249e-01, 5.4122e-06, 1.4856e-07, 1.5068e-07, 2.5381e-03, 3.3249e-01, 3.3249e-01])
#EDGE_WEIGHTS = torch.Tensor([0.0091, 0.0891, 0.8833, 0.0175, 0.0009])
EDGE_WEIGHTS = torch.Tensor([0.0091, 0.0891, 0.8833, 0.0175, 0.0091])"""


# TODO: make metric for loss. Right now does not sync correctly, I guess?
class Critic(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.reconstruction_loss = GraphReconstructionLoss()
        self.perm_loss = PermutaionMatrixPenalty()
        self.property_loss = PropertyLoss()
        self.kld_loss = KLDLoss()
        """self.element_type_recall = Recall(num_classes=11)
        self.element_type_precision = Precision(num_classes=11)
        self.element_type_accuracy = Accuracy()
        self.charge_type_recall = Recall(num_classes=5)
        self.charge_type_precision = Precision(num_classes=5)
        self.charge_type_accuracy = Accuracy()
        self.hybridization_type_recall = Recall(num_classes=7)
        self.hybridization_type_precision = Precision(num_classes=7)
        self.hybridization_type_accuracy = Accuracy()
        self.edge_recall = Recall(num_classes=5)
        self.edge_precision = Precision(num_classes=5)
        self.edge_accuracy = Accuracy()"""

    def forward(self, graph_true, graph_pred, perm, mu, logvar):
        recon_loss = self.reconstruction_loss(
            graph_true=graph_true,
            graph_pred=graph_pred
        )
        perm_loss = self.perm_loss(perm)
        property_loss = self.property_loss(
            input=graph_pred.molecular_properties[:, :-1],  # fine tune property,
            target=graph_true.molecular_properties
        )
        kld_loss = self.kld_loss(mu, logvar)
        loss = {**recon_loss, "perm_loss": perm_loss, "property_loss": property_loss, "kld_loss": kld_loss}
        #loss = {**recon_loss, "perm_loss": perm_loss}
        loss["loss"] = loss["loss"] + 0.05 * perm_loss + 0.01 * property_loss + 0.0001 * kld_loss
        #loss = recon_loss
        return loss

    def node_metrics(self, nodes_pred, nodes_true):
        element_type_true = torch.argmax(nodes_true[:, :11], axis=-1)
        element_type_pred = torch.argmax(nodes_pred[:, :11], axis=-1)
        charge_type_true = torch.argmax(nodes_true[:, 11:16], axis=-1)
        charge_type_pred = torch.argmax(nodes_pred[:, 11:16], axis=-1)
        hybridization_type_true = torch.argmax(nodes_true[:, 16:], axis=-1)
        hybridization_type_pred = torch.argmax(nodes_pred[:, 16:], axis=-1)
        metrics = {
            "element_type_recall": self.element_type_recall(
                preds=element_type_pred, target=element_type_true),
            "element_type_precision": self.element_type_precision(
                preds=element_type_pred, target=element_type_true),
            "element_type_accuracy": self.element_type_accuracy(
                preds=element_type_pred, target=element_type_true),
            "charge_type_recall": self.charge_type_recall(
                preds=charge_type_pred, target=charge_type_true),
            "charge_type_precision":  self.charge_type_precision(
                preds=charge_type_pred, target=charge_type_true),
            "charge_type_accuracy": self.charge_type_accuracy(
                preds=charge_type_pred, target=charge_type_true),
            "hybridization_type_recall": self.hybridization_type_recall(
                preds=hybridization_type_pred, target=hybridization_type_true),
            "hybridization_type_precision": self.hybridization_type_precision(
                preds=hybridization_type_pred, target=hybridization_type_true),
            "hybridization_type_accuracy": self.hybridization_type_accuracy(
                preds=hybridization_type_pred, target=hybridization_type_true)
        }

        return metrics

    def edge_metrics(self, edges_pred, edges_true):
        metrics = {
            "edge_recall": self.edge_recall(
                preds=edges_pred, target=edges_true),
            "edge_precision": self.edge_precision(
                preds=edges_pred, target=edges_true),
            "edge_accuracy": self.edge_accuracy(
                preds=edges_pred, target=edges_true),
        }
        return metrics

    def evaluate(self, graph_true, graph_pred, perm, mu, logvar, prefix=None):
        loss = self(
            graph_true=graph_true,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )
        """node_metrics = self.node_metrics(
            nodes_pred=nodes_pred,
            nodes_true=nodes_true,
        )
        edge_metrics = self.edge_metrics(
            edges_pred=edges_pred,
            edges_true=edges_true,
        )
        metrics = {**loss, **node_metrics, **edge_metrics}"""
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
        self.explicit_hydrogen_loss = CrossEntropyLoss()
        self.edge_loss = CrossEntropyLoss(weight=EDGE_WEIGHTS)

    def forward(self, graph_true, graph_pred):
        mask = graph_true.mask
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        nodes_true = graph_true.node_features[mask][:, :20]
        nodes_pred = graph_pred.node_features[mask][:, :20]
        edges_true = graph_true.edge_features[adj_mask][:, :5]
        edges_pred = graph_pred.edge_features[adj_mask][:, :5]

        element_type_pred = nodes_pred[:, :11]
        element_type_true = torch.argmax(nodes_true[:, :11], axis=-1).flatten()
        element_type_loss = self.element_type_loss(
            input=element_type_pred,
            target=element_type_true
        )
        charge_type_pred = nodes_pred[:, 11:16]
        charge_type_true = torch.argmax(nodes_true[:, 11:16], axis=-1).flatten()
        charge_type_loss = self.charge_type_loss(
            input=charge_type_pred,
            target=charge_type_true
        )
        explicit_hydrogen_pred = nodes_pred[:, 16:]
        explicit_hydrogen_true = torch.argmax(nodes_true[:, 16:], axis=-1).flatten()
        explicit_hydrogen_loss = self.explicit_hydrogen_loss(
            input=explicit_hydrogen_pred,
            target=explicit_hydrogen_true
        )

        edges_true = torch.argmax(edges_true, axis=-1).flatten()
        edge_loss = self.edge_loss(
            input=edges_pred,
            target=edges_true
        )
        node_loss = (element_type_loss + charge_type_loss + explicit_hydrogen_loss) / 3
        total_loss = node_loss + edge_loss
        loss = {
            "element_type_loss": element_type_loss,
            "charge_type_loss": charge_type_loss,
            "explicit_hydrogen_loss": explicit_hydrogen_loss,
            "edge_loss": edge_loss,
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
