import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss, MSELoss, TripletMarginLoss
from pytorch_lightning.metrics.classification import Recall, Precision, Accuracy

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
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = GraphReconstructionLoss()
        self.element_type_recall = Recall(num_classes=11)
        self.element_type_precision = Precision(num_classes=11)
        self.element_type_accuracy = Accuracy()
        self.charge_type_recall = Recall(num_classes=5)
        self.charge_type_precision = Precision(num_classes=5)
        self.charge_type_accuracy = Accuracy()
        self.hybridization_type_recall = Recall(num_classes=7)
        self.hybridization_type_precision = Precision(num_classes=7)
        self.hybridization_type_accuracy = Accuracy()
        self.mask_recall = Recall()
        self.mask_precision = Precision()
        self.mask_accuracy = Accuracy()
        self.adj_recall = Recall(num_classes=5)
        self.adj_precision = Precision(num_classes=5)
        self.adj_accuracy = Accuracy()

    def forward(self, nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred):
        loss = self.reconstruction_loss(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred
        )

        return loss

    def node_metrics(self, nodes_pred, nodes_true, mask):
        nodes_true, nodes_pred = nodes_true[mask], nodes_pred[mask]
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

    def mask_metrics(self, mask_pred, mask_true):
        mask_true = mask_true.float().flatten()
        mask_pred = mask_pred.flatten() > 0
        mask_pred = mask_pred.long()
        metrics = {
            "mask_recall": self.mask_recall(
                preds=mask_pred, target=mask_true),
            "mask_precision": self.mask_precision(
                preds=mask_pred, target=mask_true),
            "mask_accuracy": self.mask_accuracy(
                preds=mask_pred, target=mask_true),
        }
        return metrics

    def adj_metrics(self, adj_pred, adj_true, mask):
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        adj_true, adj_pred = adj_true[adj_mask], adj_pred[adj_mask]
        metrics = {
            "adj_recall": self.adj_recall(
                preds=adj_pred, target=adj_true),
            "adj_precision": self.adj_precision(
                preds=adj_pred, target=adj_true),
            "adj_accuracy": self.adj_accuracy(
                preds=adj_pred, target=adj_true),
        }
        return metrics

    def evaluate(self, nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred, prefix=None):
        loss = self(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
        )
        node_metrics = self.node_metrics(
            nodes_pred=nodes_pred,
            nodes_true=nodes_true,
            mask=mask_true
        )
        adj_metrics = self.adj_metrics(
            adj_pred=adj_pred,
            adj_true=adj_true,
            mask=mask_true
        )
        mask_metrics = self.mask_metrics(
            mask_pred=mask_pred,
            mask_true=mask_true
        )
        metrics = {**loss, **node_metrics, **adj_metrics, **mask_metrics}

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

