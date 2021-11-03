import torch
from torch.nn import BCEWithLogitsLoss, MSELoss


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
        self.edge_loss = BCEWithLogitsLoss()

    def forward(self, graph_true, graph_pred):
        mask = graph_true.mask
        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        edges_true = (graph_true.edge_features[adj_mask][:, 1] == 1).float()
        edges_pred = graph_pred.edge_features[adj_mask][:, 1]
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


class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss
