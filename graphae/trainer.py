import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDataset
from graphae.loss import critic, resemblance_loss
from torch.nn.functional import mse_loss


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)

    def forward(self, node_features, adj, mask):
        output = self.graph_ae(node_features, adj, mask)
        return output

    def prepare_data(self):
        if self.hparams["test"]:
            graphs = np.load("1000_16mnn_graphs.npy")
            self.train_dataset = MolecularGraphDataset(graphs=graphs[128:], noise=True)
            self.eval_dataset = MolecularGraphDataset(graphs=graphs[:128], noise=False)
        else:
            graphs = np.load("1000000_16mnn_graphs.npy")
            self.train_dataset = MolecularGraphDataset(graphs=graphs[self.hparams["num_eval_samples"]:], noise=True)
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
        opt_enc = torch.optim.Adam(self.graph_ae.encoder.parameters(), lr=0.00001, betas=(0.5, 0.99))
        opt_dec = torch.optim.Adam(self.graph_ae.decoder.parameters(), lr=0.00002, betas=(0.5, 0.99))
        opt_all = torch.optim.Adam(self.graph_ae.parameters(), lr=0.00001)

        """ lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=self.hparams["lr_scheduler_factor"],
            patience=self.hparams["lr_scheduler_patience"],
            cooldown=self.hparams["lr_scheduler_cooldown"]
        )
        scheduler = {
            'scheduler': lr_scheduler,
        }"""
        return [opt_enc, opt_dec], []

    def training_step(self, batch, batch_nb, optimizer_idx):
        node_features, adj, mask = batch

        output = self(node_features, adj, mask)
        # train encoder
        if optimizer_idx == 0:
            loss = mse_loss(
                input=output["mol_emb_real"],
                target=output["mol_emb_pred"]
            )
            loss = - loss
            metric = {"enc_loss": loss}

        # train decoder
        elif optimizer_idx == 1:
            loss = mse_loss(
                input=output["mol_emb_real"],
                target=output["mol_emb_pred"]
            )
            metric = {"dec_loss": loss}

        elif optimizer_idx == 2:
            loss = critic(
                mol_emb=output["mol_emb_real"],
                mol_emb_gen=output["mol_emb_pred"],
                mask=output["mask_real"],
                mask_gen=output["mask_pred"],
                adj=output["adj_real"],
                adj_gen=output["adj_pred"],
            )["total_loss"]
            metric = {"all_loss": loss}

        output = {
            "loss": loss,
            "progress_bar": metric,
            "log": metric
        }
        return output

    def validation_step(self, batch, batch_idx):
        node_features, adj, mask = batch
        output = self(node_features, adj, mask)
        metrics = critic(
            mol_emb=output["mol_emb_real"],
            mol_emb_gen=output["mol_emb_pred"],
            mask=output["mask_real"],
            mask_gen=output["mask_pred"],
            adj=output["adj_real"],
            adj_gen=output["adj_pred"]
        )
        return metrics

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        tqdm_dict = {'val_loss': out["total_loss"]}
        return {'val_loss': out["total_loss"], 'log': out, "progress_bar": tqdm_dict}

