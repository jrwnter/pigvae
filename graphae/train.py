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


class Trainer(object):
    def __init__(self, hparams):
        self.hparams = hparams
        save_dir = hparams["save_dir"] + "/run{}/".format(hparams["id"])
        if not os.path.isdir(save_dir):
            print("Creating directory")
            os.mkdir(save_dir)
        self.save_file = os.path.join(save_dir, "save.ckpt")
        graphs = np.load("1000000_16mnn_graphs.npy")
        train_dataset = MolecularGraphDataset(graphs=graphs[hparams["num_eval_samples"]:], noise=False)
        eval_dataset = MolecularGraphDataset(graphs=graphs[:hparams["num_eval_samples"]], noise=False)
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=hparams["batch_size"],
            num_workers=hparams["num_workers"],
            shuffle=True,
            pin_memory=True
        )
        self.eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=hparams["batch_size"],
            num_workers=hparams["num_workers"],
            shuffle=False,
            pin_memory=True
        )
        self.device = "cuda:{}".format(hparams["gpus"])

        self.model = GraphAE(hparams).to(self.device)

        self.opt_ae = torch.optim.Adam(self.model.encoder.parameters(), lr=0.00002, betas=(0.5, 0.99))
        self.opt_disc = torch.optim.Adam(self.model.decoder.parameters(), lr=0.00005, betas=(0.5, 0.99))
        self.scheduler_enc = torch.optim.lr_scheduler.StepLR(self.opt_enc, 50, 0.9)
        self.scheduler_dec = torch.optim.lr_scheduler.StepLR(self.opt_dec, 50, 0.9)
        self.global_step = 0
        self.num_epochs = 100

    def train(self):
        log = {"dec_loss": [], "enc_loss": []}
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            for batch in self.train_dataloader:
                self.global_step += 1
                enc_loss, dec_loss, noise_std = self.train_step(batch)
                log["enc_loss"].append(enc_loss.item())
                log["dec_loss"].append(dec_loss.item())
                if self.global_step % 100 == 0:
                    self.evaluate(train_log=log, noise_std=noise_std)
                    self.scheduler_enc.step()
                    self.scheduler_dec.step()
                    torch.save(self.model, self.save_file)
                    log = {"dec_loss": [], "enc_loss": []}

    def train_step(self, batch):

        noise_std = 0.05 + 0.3 * 0.9 ** (self.global_step / 1000)
        node_features, adj, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        node_features, adj, mask = add_noise(node_features, adj, mask, std=0.01)
        noisy_node_features, noisy_adj, noisy_mask = add_noise(node_features, adj, mask, std=noise_std)

        self.opt_dec.zero_grad()
        self.opt_enc.zero_grad()

        # AE

        mol_emb = self.model.encoder(node_features, adj, mask)
        noisy_mol_emb = self.model.encoder(noisy_node_features, noisy_adj, noisy_mask)

        node_features_pred, adj_pred, mask_pred = self.model.decoder(mol_emb.detach())
        mol_emb_pred = self.model.encoder(node_features_pred, adj_pred, mask_pred)

        enc_loss = triplet_margin_loss(
            anchor=mol_emb,
            positive=noisy_mol_emb,
            negative=mol_emb_pred,
        )
        #real_pred = self.model.descriminator(mol_emb)
        noisy_pred = self.model.descriminator(noisy_mol_emb)
        fake_pred = self.model.descriminator(mol_emb_pred)
        real_target = torch.ones([mol_emb.size(0), 1]).to(self.device)
        fake_target = torch.zeros([mol_emb.size(0), 1]).to(self.device)
        enc_loss += 0.5 * torch.nn.functional.binary_cross_entropy(
            input=noisy_pred,
            target=real_target)
        enc_loss += 0.5 * torch.nn.functional.binary_cross_entropy(
            input=fake_pred,
            target=fake_target)

        enc_loss.backward(retain_graph=True)
        self.opt_enc.step()

        dec_loss = triplet_margin_loss(
            anchor=mol_emb.detach(),
            positive=mol_emb_pred,
            negative=noisy_mol_emb.detach()
        )
        dec_loss += torch.nn.functional.binary_cross_entropy(
            input=fake_pred,
            target=real_target)

        sparsity_loss = 0.5 * torch.min(torch.abs(mask_pred - torch.ones_like(mask_pred)),
                                        torch.abs(mask_pred - torch.zeros_like(mask_pred))).mean()
        sparsity_loss += 0.5 * torch.min(torch.abs(adj_pred - torch.ones_like(adj_pred)),
                                         torch.abs(adj_pred - torch.zeros_like(adj_pred))).mean()
        dec_loss += sparsity_loss

        dec_loss.backward()
        self.opt_dec.step()



        return enc_loss, dec_loss, noise_std

    def evaluate(self, train_log, noise_std):
        with torch.no_grad():
            metrics = []
            for batch in self.eval_dataloader:
                node_features, adj, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                output = self.model(node_features, adj, mask)
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
            out[key] = round(torch.stack([output[key] for output in metrics]).mean().item(), 3)
        for key, value in train_log.items():
            train_log[key] = round(np.mean(train_log[key]), 3)
        print({**train_log, **out, **{"noise_std": round(noise_std, 3)}})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    trainer = Trainer(args.__dict__)
    trainer.train()


