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
    save_file = os.path.join(save_dir, "save.ckpt")
    if not os.path.isdir(save_dir):
        print("Creating directory")
        os.mkdir(save_dir)
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

    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=0.00002, betas=(0.5, 0.99))
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=0.0001, betas=(0.5, 0.99))

    step = 0
    log = {"dec_loss": [], "enc_loss": [], "kld_loss_real": [], "kld_loss_pred": [], "sparsity_loss": []}
    for epoch in range(50):
        print("Epoch: ", epoch)
        for batch in train_dataloader:
            step += 1
            node_features, adj, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            opt_enc.zero_grad()

            mol_emb = model.encoder(node_features, adj, mask)
            mu = model.fc_mu(mol_emb)
            logvar = model.fc_logvar(mol_emb)
            mol_emb = reparameterize(mu, logvar)
            with torch.no_grad():
                node_features_pred, adj_pred, mask_pred = model.decoder(mol_emb)
            mol_emb_pred = model.encoder(node_features_pred, adj_pred, mask_pred)
            mu_pred = model.fc_mu(mol_emb_pred)
            logvar_pred = model.fc_logvar(mol_emb_pred)
            mol_emb_pred = reparameterize(mu_pred, logvar_pred)
            enc_loss = -1 * mse_loss(
                input=mol_emb_pred,
                target=mol_emb
            )
            kld_loss_real = 0.1 * kld_loss(mu, logvar)
            kld_loss_pred = 0.1 * kld_loss(mu_pred, logvar_pred)

            loss = enc_loss + kld_loss_real + kld_loss_pred

            loss.backward()
            opt_enc.step()


            opt_dec.zero_grad()
            with torch.no_grad():
                mol_emb = model.encoder(node_features, adj, mask)
            node_features_pred, adj_pred, mask_pred = model.decoder(mol_emb)
            mol_emb_pred = model.encoder(node_features_pred, adj_pred, mask_pred)
            mu_pred = model.fc_mu(mol_emb_pred)
            logvar_pred = model.fc_logvar(mol_emb_pred)
            mol_emb_pred = reparameterize(mu_pred, logvar_pred)
            dec_loss = mse_loss(
                input=mol_emb_pred,
                target=mol_emb
            )
            sparsity_loss = 0.1 * torch.min(torch.abs(mask_pred - torch.ones_like(mask_pred)),
                                            torch.abs(mask_pred - torch.zeros_like(mask_pred))).mean()
            sparsity_loss += 0.1 * torch.min(torch.abs(adj_pred - torch.ones_like(adj_pred)),
                                             torch.abs(adj_pred - torch.zeros_like(adj_pred))).mean()
            loss = dec_loss + sparsity_loss



            loss.backward()
            opt_dec.step()

            log["enc_loss"].append(enc_loss.item())
            log["dec_loss"].append(dec_loss.item())
            log["kld_loss_real"].append(kld_loss_real.item())
            log["kld_loss_pred"].append(kld_loss_pred.item())
            log["sparsity_loss"].append(sparsity_loss.item())

            if step % 200 == 0:
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
                print({**log, **out})
                torch.save(model, save_file)
                log = {"dec_loss": [], "enc_loss": [], "kld_loss_real": [], "kld_loss_pred": [], "sparsity_loss": []}

        return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    train(args.__dict__)


