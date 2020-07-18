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
        smiles_df = pd.read_csv(self.hparams["data_path"], nrows=self.hparams["num_rows"])
        smiles_df = smiles_df[smiles_df.num_atoms <= self.hparams["max_num_nodes"]]
        smiles_list = smiles_df.smiles.tolist()
        self.train_dataset = MolecularGraphDataset(
            smiles_list=smiles_list[self.hparams["num_eval_samples"]:],
            num_nodes=self.hparams["max_num_nodes"]
        )
        self.eval_dataset = MolecularGraphDataset(
            smiles_list=smiles_list[:self.hparams["num_eval_samples"]],
            num_nodes=self.hparams["max_num_nodes"]
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
        opt_enc = torch.optim.Adam(self.graph_ae.encoder.parameters(), lr=self.hparams.lr)
        opt_dec = torch.optim.Adam(self.graph_ae.decoder.parameters(), lr=self.hparams.lr)
        opt_all = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams.lr)

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
        return [opt_enc, opt_dec, opt_all], []

    def training_step(self, batch, batch_nb, optimizer_idx):
        node_features, adj, mask = batch
        deg_real = adj.sum(axis=1)
        adj[adj == 0] += torch.randn_like(adj[adj == 0]).abs().type_as(node_features) * 0.01
        adj[adj == 1] -= torch.randn_like(adj[adj == 1]).abs().type_as(node_features) * 0.01
        mask = mask.float()
        mask[mask == 0] += torch.randn_like(mask[mask == 0]).abs().type_as(node_features) * 0.01
        mask[mask == 1] -= torch.randn_like(mask[mask == 1]).abs().type_as(node_features) * 0.01
        output = self(node_features, adj, mask)
        # train encoder
        if optimizer_idx == 0:
            loss = mse_loss(
                input=output["mol_emb_real"],
                target=output["mol_emb_pred"].detach()
            )
            loss = 1 / loss
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

