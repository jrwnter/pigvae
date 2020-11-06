import pytorch_lightning as pl
import pandas as pd
from torch_geometric.data import DataLoader
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDatasetFromSmiles, batch_to_dense
from graphae.metrics import *
from pivae.vae import PIVAE
from pivae.ddp import MyDistributedDataParallel
from time import time


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        hparams["max_num_elements"] = hparams["max_num_nodes"]
        hparams["element_dim"] = hparams["node_dim"]
        hparams["element_emb_dim"] = hparams["node_dim"]
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)
        self.pi_ae = PIVAE(hparams)
        self.critic = Critic()
        self.tau_scheduler = TauScheduler(
            start_value=1.0,
            factor=0.75,
            step_size=1
        )

    def get_postprocess_method(self, postprocess_method):
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
        return postprocess_method

    def forward(self, graph, training, tau, postprocess_method=None):
        postprocess_method = self.get_postprocess_method(postprocess_method)
        node_embs = self.graph_ae.encode(graph=graph)
        node_embs_pred, _ = self.pi_ae(node_embs, training=training, tau=tau)
        node_logits, adj_logits, mask_logits = self.graph_ae.decoder(node_embs=node_embs_pred)
        if postprocess_method is not None:
            node_logits, adj_logits = self.postprocess_logits(
                node_logits=node_logits,
                adj_logits=adj_logits,
                method=postprocess_method,
            )
        return node_logits, adj_logits, mask_logits

    def setup(self, stage: str):
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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=10,
            cooldown=50,
            min_lr=1e-6,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_adj_acc',
            'frequency': self.hparams["eval_freq"] + 1
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        sparse_graph, dense_graph = batch[0], batch[1]
        tau = self.tau_scheduler.tau
        self.log("tau", tau)
        nodes_pred, adj_pred, mask_pred = self(
            graph=sparse_graph,
            training=True,
            tau=tau
        )
        nodes_true, adj_true, mask_true = dense_graph.x, dense_graph.adj, dense_graph.mask
        loss = self.critic(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
        )
        self.log("loss", loss["loss"])
        return loss

    def configure_ddp(self, model, device_ids):
        model = MyDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model

    def validation_step(self, batch, batch_idx):
        sparse_graph, dense_graph = batch[0], batch[1]
        tau = self.tau_scheduler.tau
        nodes_true, adj_true, mask_true = dense_graph.x, dense_graph.adj, dense_graph.mask
        nodes_pred, adj_pred, mask_pred = self(
            graph=sparse_graph,
            training=False,
            tau=tau
        )
        metrics = self.critic.evaluate(
            nodes_true=nodes_true,
            adj_true=adj_true,
            mask_true=mask_true,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred,
            mask_pred=mask_pred,
            prefix="val"
        )
        return metrics

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        for metric, value in out.items():
            self.log(metric, value)

    def on_epoch_end(self):
        self.tau_scheduler()


class TauScheduler(object):
    def __init__(self, start_value, factor, step_size):
        self.tau = start_value
        self.factor = factor
        self.step_size = step_size
        self.steps = 0

    def __call__(self):
        self.steps += 1
        if self.steps >= self.step_size:
            self.tau *= self.factor
            self.steps = 0
