import numpy as np
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE, reparameterize
from graphae.data import MolecularGraphDataset, add_noise
from graphae.loss import critic, resemblance_loss, kld_loss
from graphae.hyperparameter import add_arguments
from torch.nn.functional import mse_loss, triplet_margin_loss


def train(hparams):
    save_dir = hparams["save_dir"] + "/run{}/".format(hparams["id"])
    if not os.path.isdir(save_dir):
        print("Creating directory")
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, "save.ckpt")
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
    device = "cuda:{}".format(hparams["gpus"])

    model = GraphAE(hparams).to(device)

    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=0.00002, betas=(0.8, 0.99))
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=0.00005, betas=(0.8, 0.99))

    step = 0
    log = {"dec_loss": [], "enc_loss": []}
    bla = 1
    for epoch in range(5000):
        print("Epoch: ", epoch)
        for batch in train_dataloader:
            step += 1
            noise_std = 0.05 + 0.3 * 0.9 ** (step / 1000)
            node_features, adj, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            node_features, adj, mask = add_noise(node_features, adj, mask, std=0.01)
            noisy_node_features, noisy_adj, noisy_mask = add_noise(node_features, adj, mask, std=noise_std)
            opt_dec.zero_grad()
            mol_emb = model.encoder(node_features, adj, mask)
            noisy_mol_emb = model.encoder(noisy_node_features, noisy_adj, noisy_mask)
            node_features_pred, adj_pred, mask_pred = model.decoder(mol_emb.detach())
            mol_emb_pred = model.encoder(node_features_pred, adj_pred, mask_pred)
            dec_loss = triplet_margin_loss(
                anchor=mol_emb.detach(),
                positive=mol_emb_pred,
                negative=noisy_mol_emb.detach()
            )
            loss = dec_loss

            loss.backward()
            opt_dec.step()


            opt_enc.zero_grad()

            mol_emb = model.encoder(node_features, adj, mask)
            noisy_mol_emb = model.encoder(noisy_node_features, noisy_adj, noisy_mask)
            with torch.no_grad():
                node_features_pred, adj_pred, mask_pred = model.decoder(mol_emb)
            mol_emb_pred = model.encoder(node_features_pred, adj_pred, mask_pred)
            enc_loss = triplet_margin_loss(
                anchor=mol_emb,
                positive=noisy_mol_emb,
                negative=mol_emb_pred,
            )

            loss = enc_loss

            loss.backward()
            opt_enc.step()

            log["enc_loss"].append(enc_loss.item())
            log["dec_loss"].append(dec_loss.item())

            if step % 50 == 0:
                metrics = []
                for batch in eval_dataloader:
                    node_features, adj, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    output = model(node_features, adj, mask)
                    metrics.append(critic(
                        mol_emb=output["mol_emb_real"],
                        mol_emb_gen=output["mol_emb_pred"],
                        mask=output["mask_real"],
                        mask_gen=output["mask_pred"],
                        adj=output["adj_real"],
                        adj_gen=output["adj_pred"],
                    ))
                out = {}
                for key in metrics[0].keys():
                    out[key] = torch.stack([output[key] for output in metrics]).mean().item()
                for key, value in log.items():
                    log[key] = np.mean(log[key])
                print({**log, **out, **{"noise_std": noise_std}})
                torch.save(model, save_file)
                log = {"dec_loss": [], "enc_loss": []}
                bla = 0

        return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    train(args.__dict__)


