import numpy as np
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphVAEGAN, postprocess
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
        #toy_example = np.stack(10000 * [graphs[0]])
        #train_dataset = MolecularGraphDataset(graphs=toy_example[128:], noise=False)
        #eval_dataset = MolecularGraphDataset(graphs=toy_example[:128], noise=False)
        train_dataset = MolecularGraphDataset(graphs=graphs[1024:], noise=False)
        eval_dataset = MolecularGraphDataset(graphs=graphs[:1024], noise=False)
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

        self.model = GraphVAEGAN(hparams).to(self.device)

        self.opt_g = torch.optim.Adam(list(self.model.generator.parameters()) + list(self.model.encoder.parameters()) + list(self.model.linear_mu.parameters()) + list(self.model.linear_logvar.parameters()), lr=0.00002, betas=(0.5, 0.99))
        self.opt_d = torch.optim.Adam(list(self.model.discriminator.parameters()) + list(self.model.encoder.parameters()), lr=0.00002, betas=(0.5, 0.99))
        self.opt_e = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.linear_mu.parameters()) + list(self.model.linear_logvar.parameters()), lr=0.00002, betas=(0.5, 0.99))
        self.global_step = 0
        self.num_epochs = 10000

        self.criterion = torch.nn.BCELoss()

    def train(self):
        log = {"g_loss": [], "d_loss": [], "e_loss_kld": [], "g_loss_rec": []}
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            for batch in self.train_dataloader:
                self.global_step += 1
                if self.global_step % 5 == 0:
                    d_loss, g_loss, e_loss_kld, g_loss_rec = self.train_step(batch, train_gen=True)
                    log["g_loss"].append(g_loss.item())
                    log["g_loss_rec"].append(g_loss_rec.item())
                else:
                    d_loss, e_loss_kld = self.train_step(batch, train_gen=False)
                log["d_loss"].append(d_loss.item())
                log["e_loss_kld"].append(e_loss_kld.item())
                if self.global_step % 100 == 0:
                    self.evaluate(train_log=log)
                    torch.save(self.model, self.save_file)
                    log = {"g_loss": [], "d_loss": [], "e_loss_kld": [], "g_loss_rec": []}


    def train_step(self, batch, train_gen):
        nodes, adj, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        adj_ = torch.FloatTensor(adj.shape.numel(), 2).to(self.device).zero_().scatter_(1, adj.view(-1, 1).long(), 1).view(list(adj.shape) + [2])

        z_pri = torch.randn([adj.shape[0], self.hparams["emb_dim"]]).to(self.device)

        # Disciminator

        self.opt_d.zero_grad()
        with torch.no_grad():
            z_enc, _, _ = self.model.encode(nodes, adj)
            nodes_fake_enc, adj_fake_enc, nodes_fake_enc_, adj_fake_enc_ = self.model.generate(z_enc)
            nodes_fake_pri, adj_fake_pri, nodes_fake_pri_, adj_fake_pri_ = self.model.generate(z_pri)

        d_logits_real, _ = self.model.discriminate(nodes, adj)
        d_logits_fake_enc, _ = self.model.discriminate(nodes_fake_enc, adj_fake_enc)
        d_logits_fake_pri, _ = self.model.discriminate(nodes_fake_pri, adj_fake_pri)

        eps = torch.rand(d_logits_real.size(0), 1, 1).to(self.device)
        x_enc1 = (eps * nodes + (1. - eps) * nodes_fake_enc_).requires_grad_(True)
        x_enc2 = (eps.unsqueeze(-1) * adj_ + (1. - eps.unsqueeze(-1)) * adj_fake_enc_).requires_grad_(True)
        x_pri1 = (eps * nodes + (1. - eps) * nodes_fake_pri_).requires_grad_(True)
        x_pri2 = (eps.unsqueeze(-1) * adj_ + (1. - eps.unsqueeze(-1)) * adj_fake_pri_).requires_grad_(True)

        x_enc1, x_enc2 = postprocess(x_enc1, "hard_gumbel"), postprocess(x_enc2, "hard_gumbel")[:, :, :, 1]
        x_pri1, x_pri2 = postprocess(x_pri1, "hard_gumbel"), postprocess(x_pri2, "hard_gumbel")[:, :, :, 1]
        grad_enc, _ = self.model.discriminate(x_enc1, x_enc2)
        grad_pri, _ = self.model.discriminate(x_pri1, x_pri2)

        d_loss_gp = self.gradient_penalty(grad_pri, x_pri1) + self.gradient_penalty(grad_pri, x_pri2) + self.gradient_penalty(grad_enc, x_enc1) + self.gradient_penalty(grad_enc, x_enc2)

        d_loss = -torch.mean(d_logits_real) + 0.5 * torch.mean(d_logits_fake_enc) + 0.5 * torch.mean(d_logits_fake_pri) + 10 * d_loss_gp
        d_loss.backward()
        self.opt_d.step()

        # Encoder

        self.opt_e.zero_grad()
        z_enc, mu, log_var = self.model.encode(nodes, adj)

        e_loss_kld = kld_loss(mu, log_var)

        e_loss = 0.1 * e_loss_kld

        e_loss.backward()
        self.opt_e.step()

        # Generator
        self.opt_g.zero_grad()

        if train_gen:
            z_enc, _, _ = self.model.encode(nodes, adj)

            nodes_fake_enc, adj_fake_enc, nodes_fake_enc_, adj_fake_enc_ = self.model.generate(z_enc)
            nodes_fake_enc_d, adj_fake_enc_d, nodes_fake_enc_d_, adj_fake_enc_d_ = self.model.generate(z_enc.detach())
            nodes_fake_pri, adj_fake_pri, nodes_fake_pri_, adj_fake_pri_ = self.model.generate(z_pri)

            _, d_features_real = self.model.discriminate(nodes, adj)
            _, d_features_fake_enc = self.model.discriminate(nodes_fake_enc, adj_fake_enc)
            d_logits_fake_enc_d, _ = self.model.discriminate(nodes_fake_enc_d, adj_fake_enc_d)
            d_logits_fake_pri, _ = self.model.discriminate(nodes_fake_pri, adj_fake_pri)

            g_loss_gen = -0.5 * (torch.mean(d_logits_fake_enc_d) + torch.mean(d_logits_fake_pri))
            batch_size = d_features_real.size(0)
            norm_factor = torch.triu(
                ((d_features_real.unsqueeze(0) - d_features_real.unsqueeze(1)) ** 2).mean(dim=-1)).sum() / (
                                      (batch_size * batch_size - 1) / 2)
            g_loss_rec = mse_loss(d_features_fake_enc, d_features_real) / norm_factor
            g_loss = g_loss_gen + g_loss_rec

            g_loss.backward()
            self.opt_g.step()
            return d_loss, g_loss, e_loss_kld, g_loss_rec
        return d_loss, e_loss_kld

    def evaluate(self, train_log):
        """with torch.no_grad():
            metrics = []
            for batch in self.eval_dataloader:
                node_features, adj, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                node_features_pred, adj_pred, mask_pred = self.model(node_features, adj, mask)
                metrics.append(critic(
                    mask=mask,
                    mask_gen=mask_pred,
                    adj=adj,
                    adj_gen=adj_pred,
                ))
        out = {}
        for key in metrics[0].keys():
            out[key] = round(torch.stack([output[key] for output in metrics]).mean().item(), 3)"""
        for key, value in train_log.items():
            train_log[key] = round(np.mean(train_log[key]), 3)
        print({**train_log})

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device).requires_grad_(True)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = ((dydx.norm(dim=1) - 1) ** 2).mean()
        #dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        #return torch.mean((dydx_l2norm - 1) ** 2)
        return dydx_l2norm





if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    trainer = Trainer(args.__dict__)
    trainer.train()


