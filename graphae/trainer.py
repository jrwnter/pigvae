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
        self.tf_scheduler = TeacherForcingScheduler(start_value=1.0, factor=0.8)

    def forward(self, graph, teacher_forcing, postprocess_method=None):
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
        node_logits, adj_logits, mask_logits, node_emb_enc, node_emb_dec = self.graph_ae(
            graph=graph,
            teacher_forcing=teacher_forcing,
            postprocess_method=postprocess_method,
            postprocess_temp=self.hparams["postprocess_temp"]
        )
        return node_logits, adj_logits, mask_logits, node_emb_enc, node_emb_dec

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
            patience=10,
            cooldown=30,
            min_lr=1e-6,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.hparams["eval_freq"] + 1
        }"""
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1,
            gamma=0.8,
        )
        scheduler = {
            'scheduler': lr_scheduler,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        sparse_graph, dense_graph = batch[0], batch[1]
        tf_prop = self.tf_scheduler.tf_prop
        nodes_pred, adj_pred, mask_pred, node_emb_enc, node_emb_dec = self(sparse_graph, teacher_forcing=tf_prop)
        nodes_true, adj_true, mask_true = dense_graph.x, dense_graph.adj, dense_graph.mask
        loss = self.critic(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
            node_emb_enc=node_emb_enc,
            node_emb_dec=node_emb_dec,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sparse_graph, dense_graph = batch[0], batch[1]
        nodes_true, adj_true, mask_true = dense_graph.x, dense_graph.adj, dense_graph.mask
        nodes_pred, adj_pred, mask_pred, node_emb_enc, node_emb_dec = self(
            graph=sparse_graph,
            teacher_forcing=True,
            postprocess_method=None
        )
        metrics_tf = self.critic.evaluate(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
            node_emb_enc=node_emb_enc,
            node_emb_dec=node_emb_dec,
        )
        nodes_pred, adj_pred, mask_pred, node_emb_enc, node_emb_dec = self(
            graph=sparse_graph,
            teacher_forcing=0.0,
            postprocess_method=None
        )

        metrics_no_tf = self.critic.evaluate(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
            node_emb_enc=node_emb_enc,
            node_emb_dec=node_emb_dec,
            prefix="no_tf"
        )
        metrics = {**metrics_tf, **metrics_no_tf}
        return metrics

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        tqdm_dict = {'val_loss': out["loss"]}

        return {'val_loss': out["loss"], 'log': out, "progress_bar": tqdm_dict}

    def on_epoch_end(self):
        self.tf_scheduler()


class TeacherForcingScheduler(object):
    def __init__(self, start_value, factor, step_size):
        self.tf_prop = start_value
        self.factor = factor
        self.step_size = step_size
        self.steps = 0

    def __call__(self):
        self.steps += 1
        if self.step_size >= self.steps:
            self.tf_prop *= self.factor






