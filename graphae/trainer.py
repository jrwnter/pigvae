import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDataset, add_noise
from graphae.loss import critic, resemblance_loss, kld_loss
from torch.nn.functional import mse_loss, triplet_margin_loss


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
            self.train_dataset = MolecularGraphDataset(graphs=graphs[128:], noise=False)
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
        opt_enc = torch.optim.Adam(self.graph_ae.encoder.parameters(), lr=0.00002, betas=(0.5, 0.99))
        opt_dec = torch.optim.Adam(self.graph_ae.decoder.parameters(), lr=0.0001, betas=(0.5, 0.99))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=opt_enc,
            step_size=2,
            gamma=0.5
        )
        scheduler_enc = {
            'scheduler': lr_scheduler,
        }
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=opt_dec,
            step_size=2,
            gamma=0.5
        )
        scheduler_dec = {
            'scheduler': lr_scheduler,
        }
        return [opt_dec, opt_enc], [scheduler_enc, scheduler_dec]

    def training_step(self, batch, batch_nb, optimizer_idx):
        node_features, adj, mask = batch
        noisy_node_features, noisy_adj, noisy_mask = add_noise(node_features, adj, mask,
                                                               std=0.2 * (0.9 ** self.current_epoch))
        # train decoder
        if optimizer_idx == 0:
            mol_emb = self.graph_ae.encoder(node_features, adj, mask).detach()
            node_features_pred, adj_pred, mask_pred = self.graph_ae.decoder(mol_emb)
            mol_emb_pred = self.graph_ae.encoder(node_features_pred, adj_pred, mask_pred)
            noisy_mol_emb_real = self.graph_ae.encoder(noisy_node_features, noisy_adj, noisy_mask).detach()
            loss = triplet_margin_loss(
                anchor=mol_emb,
                positive=mol_emb_pred,
                negative=noisy_mol_emb_real,
                margin=0.5
            )
            loss += 0.1 * torch.min(torch.abs(mask_pred - torch.ones_like(mask_pred)),
                                    torch.abs(mask_pred - torch.zeros_like(mask_pred))).mean()
            loss += 0.1 * torch.min(torch.abs(adj_pred - torch.ones_like(adj_pred)),
                                    torch.abs(adj_pred - torch.zeros_like(adj_pred))).mean()
            metric = {"dec_loss": loss}

        # train encoder
        elif optimizer_idx == 1:
            mol_emb = self.graph_ae.encoder(node_features, adj, mask)
            node_features_pred, adj_pred, mask_pred = self.graph_ae.decoder(mol_emb)
            node_features_pred = node_features_pred.detach()
            adj_pred = adj_pred.detach()
            mask_pred = mask_pred.detach()
            mol_emb_pred = self.graph_ae.encoder(node_features_pred, adj_pred, mask_pred)
            noisy_mol_emb_real = self.graph_ae.encoder(noisy_node_features, noisy_adj, noisy_mask)
            loss = triplet_margin_loss(
                anchor=mol_emb,
                positive=noisy_mol_emb_real,
                negative=mol_emb_pred,
                margin=0.5
            )
            metric = {"enc_loss": loss}

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
            adj_gen=output["adj_pred"],
        )
        return metrics

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        tqdm_dict = {'val_loss': out["total_loss"]}
        return {'val_loss': out["total_loss"], 'log': out, "progress_bar": tqdm_dict}

