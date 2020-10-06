import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDataset, add_noise
from graphae.metrics import *
from torch.nn.functional import mse_loss, triplet_margin_loss


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)
        self.critic = GraphReconstructionLoss(num_node_types=11, num_edge_types=1)

    def forward(self, node_features, adj, mask):
        node_features, adj = self.graph_ae(node_features, adj, mask)
        return node_features, adj

    def prepare_data(self):
        if self.hparams["test"]:
            graphs = np.load("1000_16mnn_graphs.npy")
            self.train_dataset = MolecularGraphDataset(graphs=graphs[128:], noise=False)
            self.eval_dataset = MolecularGraphDataset(graphs=graphs[:128], noise=False)
        else:
            graphs = np.load("1000000_16mnn_graphs.npy")
            self.train_dataset = MolecularGraphDataset(graphs=graphs[self.hparams["num_eval_samples"]:], noise=False)
            self.eval_dataset = MolecularGraphDataset(graphs=graphs[:self.hparams["num_eval_samples"]], noise=False)

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
        optimizer = torch.optim.Adam(self.graph_ae.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.5
        )
        scheduler = {
            'scheduler': lr_scheduler,
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        nodes, adj, mask = batch
        nodes_pred, adj_pred = self(nodes, adj, mask)
        node_loss, adj_loss = self.critic(
            nodes_true=nodes,
            adj_true=adj,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred
        )
        output = {
            "node_loss": node_loss,
            "adj_loss": adj_loss,
            "loss": node_loss + adj_loss
        }
        return output

    def validation_step(self, batch, batch_idx):
        nodes, adj, mask = batch
        nodes_pred, adj_pred = self(nodes, adj, mask)
        node_loss, adj_loss = self.critic(
            nodes_true=nodes,
            adj_true=adj,
            nodes_pred=nodes_pred,
            adj_pred=adj_pred
        )
        nodes_pred_oh, adj_pred_oh = self.graph_ae.logits_to_one_hot(nodes_pred, adj_pred)
        node_acc = node_balanced_accuracy(input=nodes_pred_oh, target=nodes)
        adj_acc = adj_balanced_accuracy(input=adj_pred_oh, target=adj)
        output = {
            "node_loss": node_loss,
            "adj_loss": adj_loss,
            "loss": node_loss + adj_loss,
            "node_acc": node_acc,
            "adj_acc": adj_acc,
        }
        return output

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        tqdm_dict = {'val_loss': out["loss"], "node_acc": out["node_acc"], "adj_acc": out["adj_acc"]}
        return {'val_loss': out["loss"], 'log': out, "progress_bar": tqdm_dict}

