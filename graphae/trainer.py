import pytorch_lightning as pl
import pandas as pd
from torch_geometric.data import DataLoader
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDatasetFromSmiles, batch_to_dense
from graphae.metrics import *
from time import time


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        if "alpha" not in hparams:
            hparams["alpha"] = 0.01
        if "postprocess_method" not in hparams:
            hparams["postprocess_method"] = 0
        if "postprocess_temp" not in hparams:
            hparams["postprocess_temp"] = 1.0
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)
        self.critic = Critic(alpha=hparams["alpha"])
        self.alpha_decay = AlphaDecay(hparams["alpha"], 0.99, 2, cooldown=20, patience=5)

    def forward(self, graph, permute=True, round_perm=False, postprocess_method=None):
        if postprocess_method is None:
            if self.hparams["postprocess_method"] == 0:
                postprocess_method = None
            elif self.hparams["postprocess_method"] == 1:
                postprocess_method = "soft_gumbel"
            elif self.hparams["postprocess_method"] == 2:
                postprocess_method = "hard_gumbel"
            elif self.hparams["postprocess_method"] == 3:
                postprocess_method = "softmax"
            else:
                raise NotImplementedError
        node_logits, adj_logits, mask_logits, perms = self.graph_ae(
            graph=graph,
            permute=permute,
            round_perm=round_perm,
            postprocess_method=postprocess_method,
            postprocess_temp=self.hparams["postprocess_temp"]
        )
        return node_logits, adj_logits, mask_logits, perms

    def prepare_data(self):
        num_smiles = 1000000 if self.hparams["test"] else None
        smiles_df = pd.read_csv("smiles_{}_atoms.csv".format(self.hparams["max_num_nodes"]), nrows=num_smiles)
        self.train_dataset = MolecularGraphDatasetFromSmiles(
            smiles_list=smiles_df.iloc[self.hparams["num_eval_samples"]:].smiles.tolist(),
            num_nodes=self.hparams["max_num_nodes"],
        )
        self.eval_dataset = MolecularGraphDatasetFromSmiles(
            smiles_list=smiles_df.iloc[:self.hparams["num_eval_samples"]].smiles.tolist(),
            num_nodes=self.hparams["max_num_nodes"],
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"])
        """lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=5,
            cooldown=20,
            min_lr=1e-6,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.hparams["eval_freq"] + 1
        }"""
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.5,
        )
        scheduler = {
            'scheduler': lr_scheduler,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        sparse_graph, dense_graph = batch[0], batch[1]
        nodes_pred, adj_pred, mask_pred, perm = self(sparse_graph)
        nodes_true, adj_true, mask_true = dense_graph.x, dense_graph.adj, dense_graph.mask
        loss = self.critic(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
            perm=perm,
            alpha=self.alpha_decay.alpha
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sparse_graph, dense_graph = batch[0], batch[1]
        nodes_true, adj_true, mask_true = dense_graph.x, dense_graph.adj, dense_graph.mask
        nodes_pred, adj_pred, mask_pred, perm = self(
            graph=sparse_graph,
            permute=False,
            postprocess_method=None
        )
        nodes_pred_, adj_pred_, mask_pred_, perm_ = self.graph_ae.permute(
            nodes=nodes_pred,
            adj=adj_pred,
            mask=mask_pred,
            perm=perm,
            round=False
        )
        metrics_soft = self.critic.evaluate(nodes_true, adj_true, mask_true, nodes_pred_, adj_pred_, mask_pred_, perm, self.alpha_decay.alpha)
        nodes_pred, adj_pred = self.graph_ae.postprocess_logits(
            node_logits=nodes_pred,
            adj_logits=adj_pred,
            method="softmax"
        )
        nodes_pred, adj_pred, mask_pred, perm_ = self.graph_ae.permute(
            nodes=nodes_pred,
            adj=adj_pred,
            mask=mask_pred,
            perm=perm,
            round=True
        )
        metrics_hard = self.critic.evaluate(nodes_true, adj_true, mask_true, nodes_pred, adj_pred, mask_pred, perm, self.alpha_decay.alpha, "hard")
        metrics = {**metrics_soft, **metrics_hard}
        return metrics

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        tqdm_dict = {'val_loss': out["loss"]}

        #self.alpha_decay(out["adj_acc"])
        out["alpha"] = self.alpha_decay.alpha

        return {'val_loss': out["loss"], 'log': out, "progress_bar": tqdm_dict}


class AlphaDecay(object):
    def __init__(self, start_alpha, target_metric_value, factor, cooldown=0, patience=0):
        self.alpha = start_alpha
        self.factor = factor
        self.cooldown = cooldown
        self.patience = patience
        self.target_metric_value = target_metric_value
        self.num_steps_below = 0
        self.steps_sice_decay = 0

    def __call__(self, metric):
        self.steps_sice_decay += 1
        if metric >= self.target_metric_value:
            self.num_steps_below += 1
            if self.steps_sice_decay >= self.cooldown:
                if self.num_steps_below >= self.patience:
                    self.alpha *= self.factor
                    self.steps_sice_decay = 0
        else:
            self.num_steps_below = 0






