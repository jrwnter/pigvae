import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDataset
from graphae.loss import critic


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)

    def forward(self, node_features, adj, mask):
        mol_emb, mol_emb_gen, adj_gen, mask_gen, node_features_gen = self.graph_ae(node_features, adj, mask)
        return mol_emb, mol_emb_gen, adj_gen, mask_gen, node_features_gen

    def prepare_data(self):
        smiles_df = pd.read_csv(self.hparams["data_path"])
        smiles_df = smiles_df[smiles_df.num_atoms <= self.hparams["max_num_nodes"]]
        smiles_list = smiles_df.smiles.tolist()
        self.train_dataset = MolecularGraphDataset(
            smiles_list=smiles_list[1024:],
            num_nodes=self.hparams["max_num_nodes"]
        )
        self.eval_dataset = MolecularGraphDataset(
            smiles_list=smiles_list[:1024],
            num_nodes=self.hparams["max_num_nodes"]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=32, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=64, num_workers=32, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=self.hparams["lr_scheduler_factor"],
            patience=self.hparams["lr_scheduler_patience"],
            cooldown=self.hparams["lr_scheduler_cooldown"]
        )
        scheduler = {
            'scheduler': lr_scheduler,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        node_features, adj, mask = batch
        mol_emb, mol_emb_gen, adj_gen, mask_gen, node_features_gen = self(node_features, adj, mask)
        loss = critic(mol_emb, mol_emb_gen, mask, mask_gen, adj, adj_gen)
        return {'loss': loss["total_loss"]}

    def validation_step(self, batch, batch_idx):
        node_features, adj, mask = batch
        mol_emb, mol_emb_gen, adj_gen, mask_gen, node_features_gen = self(node_features, adj, mask)
        loss = critic(mol_emb, mol_emb_gen, mask, mask_gen, adj, adj_gen)
        return loss

    def validation_epoch_end(self, outputs):
        out = {}
        for key in outputs[0].keys():
            out[key] = torch.stack([output[key] for output in outputs]).mean()
        tqdm_dict = {'val_loss': out["total_loss"]}
        return {'val_loss': out["total_loss"], 'log': out, "progress_bar": tqdm_dict}
