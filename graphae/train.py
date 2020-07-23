import numpy as np
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE
from graphae.data import MolecularGraphDataset, add_noise
from graphae.loss import critic, resemblance_loss, kld_loss
from graphae.hyperparameter import add_arguments
from torch.nn.functional import mse_loss, triplet_margin_loss


def train(hparams):
    graphs = np.load("1000000_16mnn_graphs.npy")
    train_dataset = MolecularGraphDataset(graphs=graphs[hparams["num_eval_samples"]:], noise=True)
    eval_dataset = MolecularGraphDataset(graphs=graphs[:hparams["num_eval_samples"]], noise=False)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        shuffle=True,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        shuffle=False,
        pin_memory=True
    )
    device = "cuda:{}".fomrat(hparams["gpus"])

    model = GraphAE(hparams)

    opt_enc = torch.optim.Adam(list(model.encoder.parameters()) + list(model.predictor.parameters()),
                               lr=0.0001)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=0.0001)

    for epoch in range(50):
        for batch in train_dataloader:
            node_features, adj, mask = batch[0].to(device)
            noisy_node_features, noisy_adj, noisy_mask = add_noise(
                node_features, adj, mask, std=0.2 * (0.9 ** epoch))

            opt_enc.zero_grad()

            mol_emb = model.encoder(node_features, adj, mask)
            with torch.no_grad():
                node_features_pred, adj_pred, mask_pred = model.decoder(mol_emb)
            mol_emb_pred = model.encoder(node_features_pred, adj_pred, mask_pred)
            noisy_mol_emb_real = model.encoder(noisy_node_features, noisy_adj, noisy_mask)
            enc_loss = triplet_margin_loss(
                anchor=mol_emb,
                positive=noisy_mol_emb_real,
                negative=mol_emb_pred,
                margin=0.5
            )



if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    train(args.__dict__)


