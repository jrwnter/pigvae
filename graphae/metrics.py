import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

# 32
ELEMENT_TYPE_WEIGHTS = torch.Tensor([
    1.1111e-04, 6.9739e-04, 7.9399e-04, 4.8409e-03, 5.1567e-03, 2.6155e-01,
    2.7761e-01, 8.4134e-03, 1.8924e-02, 1.4429e-01, 2.7761e-01])
CHARGE_TYPE_WEIGHTS = torch.Tensor([0.2596, 0.2596, 0.2205, 0.2596, 0.0006])
EDGE_WEIGHTS = torch.Tensor([1, 10, 100, 1, 1])


# TODO: make metric for loss. Right now does not sync correctly, I guess?
class Critic(torch.nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.reconstruction_loss = GraphReconstructionLoss()
        self.perm_loss = PermutaionMatrixPenalty()
        self.eos_loss = EndOfSequenceLoss()
        self.kld_loss = KLDLoss()

    def forward(self, nodes_true, edges_true, nodes_pred, edges_pred, perm, mask, eos, mu, logvar):
        recon_loss = self.reconstruction_loss(
            nodes_true=nodes_true,
            edges_true=edges_true,
            nodes_pred=nodes_pred,
            edges_pred=edges_pred,
        )
        perm_loss = self.perm_loss(perm)
        eos_loss = self.eos_loss(eos, mask)
        kld_loss = self.kld_loss(mu, logvar)
        loss = {
            **recon_loss,
            "perm_loss": perm_loss,
            "eos_loss": eos_loss,
            "kld_loss": kld_loss
        }
        loss["loss"] = loss["loss"] + self.alpha * kld_loss
        return loss

    def evaluate(self, nodes_true, edges_true, nodes_pred, edges_pred, perm, mask, eos, mu, logvar, prefix=None):
        loss = self(
            nodes_true=nodes_true,
            edges_true=edges_true,
            nodes_pred=nodes_pred,
            edges_pred=edges_pred,
            mask=mask,
            eos=eos,
            mu=mu,
            logvar=logvar,
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
        self.element_type_loss = CrossEntropyLoss(weight=ELEMENT_TYPE_WEIGHTS)
        self.charge_type_loss = CrossEntropyLoss(weight=CHARGE_TYPE_WEIGHTS)
        self.explicit_hydrogen_loss = CrossEntropyLoss()
        self.edge_loss = CrossEntropyLoss(weight=EDGE_WEIGHTS)

    def forward(self, nodes_true, edges_true, nodes_pred, edges_pred):
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


class EndOfSequenceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eos_loss = BCEWithLogitsLoss()

    def forward(self, eos_pred, mask):
        eos_true = ~ mask
        eos_true = eos_true.float()
        loss = self.eos_loss(
            input=eos_pred.flatten(),
            target=eos_true.flatten()
        )
        return loss


class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
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
    explicit_hydrogen_true, explicit_hydrogen_pred = nodes_true[:, 16:], nodes_pred[:, 16:]
    element_type_acc = scipy_balanced_accuracy(element_type_pred, element_type_true)
    charge_type_acc = scipy_balanced_accuracy(charge_type_pred, charge_type_true)
    explicit_hydrogen_acc = scipy_balanced_accuracy(explicit_hydrogen_pred, explicit_hydrogen_true)
    return element_type_acc, charge_type_acc, explicit_hydrogen_acc


def edge_balanced_accuracy(edges_pred, edges_true):
    acc = scipy_balanced_accuracy(edges_pred, edges_true)
    return acc
