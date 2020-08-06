import numpy as np
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphVAEGAN, Encoder, Descriminator, Decoder, reparameterize
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

        self.opt_g = torch.optim.Adam(list(self.model.generator.parameters()) + list(self.model.encoder.parameters()), lr=0.00005, betas=(0.5, 0.99))
        self.opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.99))
        self.global_step = 0
        self.num_epochs = 10000

        self.criterion = torch.nn.BCELoss()

    def train(self):
        log = {"gen_loss": [], "disc_loss": []}
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            for batch in self.train_dataloader:
                self.global_step += 1
                disc_loss, gen_loss = self.train_step(batch, train_generator=True)
                log["disc_loss"].append(disc_loss.item())
                log["gen_loss"].append(gen_loss.item())
                if self.global_step % 100 == 0:
                    self.evaluate(train_log=log)
                    torch.save(self.model, self.save_file)
                    log = {"gen_loss": [], "disc_loss": []}

    def train_step(self, batch, train_generator):
        nodes, adj, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        real_target = torch.ones(nodes.size(0)).to(self.device)
        fake_target = torch.zeros(nodes.size(0)).to(self.device)
        z_pri = torch.randn([adj.shape[0], self.hparams["emb_dim"]]).to(self.device)

        # Disciminator

        self.opt_d.zero_grad()

        with torch.no_grad():
            z_enc = self.model.encoder(nodes, adj)
            mu, log_var = z_enc[:, :self.hparams["emb_dim"]], z_enc[:, self.hparams["emb_dim"]:]
            z_enc = reparameterize(mu, log_var)

            nodes_fake_enc, adj_fake_enc = self.model.generator(z_enc)
            nodes_fake_enc, adj_fake_enc = postprocess(nodes_fake_enc, "hard_gumbel"), postprocess(adj_fake_enc, "hard_gumbel")
            adj_fake_enc = adj_fake_enc[:, :, :, 1]
            mask_fake_enc = nodes_fake_enc[:, :, -1] == 0
            nodes_fake_enc = nodes_fake_enc[:, :, :-1]

            nodes_fake_pri, adj_fake_pri = self.model.generator(z_pri)
            nodes_fake_pri, adj_fake_pri = postprocess(nodes_fake_pri, "hard_gumbel"), postprocess(adj_fake_pri, "hard_gumbel")
            adj_fake_pri = adj_fake_pri[:, :, :, 1]
            mask_fake_pri = nodes_fake_pri[:, :, -1] == 0
            nodes_fake_pri = nodes_fake_pri[:, :, :-1]

        d_logits_real, d_features_real = self.model.discriminator(nodes, adj, None)
        d_logits_fake_enc, d_features_fake_enc = self.model.discriminator(nodes_fake_enc, adj_fake_enc, None)
        d_logits_fake_pri, d_features_fake_pri = self.model.discriminator(nodes_fake_pri, adj_fake_pri, None)

        d_loss = self.criterion(d_logits_real, real_target) + self.criterion(d_logits_fake_enc, fake_target) + self.criterion(d_logits_fake_pri, fake_target)
        d_loss.backward()
        self.opt_d.step()


        # Generator
        self.opt_g.zero_grad()

        z_enc = self.model.encoder(nodes, adj)
        mu, log_var = z_enc[:, :self.hparams["emb_dim"]], z_enc[:, self.hparams["emb_dim"]:]
        z_enc = reparameterize(mu, log_var)

        nodes_fake_enc, adj_fake_enc = self.model.generator(z_enc)
        nodes_fake_enc, adj_fake_enc = postprocess(nodes_fake_enc, "hard_gumbel"), postprocess(adj_fake_enc,
                                                                                               "hard_gumbel")
        adj_fake_enc = adj_fake_enc[:, :, :, 1]
        mask_fake_enc = nodes_fake_enc[:, :, -1] == 0
        nodes_fake_enc = nodes_fake_enc[:, :, :-1]

        nodes_fake_enc_d, adj_fake_enc_d = self.model.generator(z_enc.detach())
        nodes_fake_enc_d, adj_fake_enc_d = postprocess(nodes_fake_enc_d, "hard_gumbel"), postprocess(adj_fake_enc_d,
                                                                                               "hard_gumbel")
        adj_fake_enc_d = adj_fake_enc_d[:, :, :, 1]
        mask_fake_enc_d = nodes_fake_enc_d[:, :, -1] == 0
        nodes_fake_enc_d = nodes_fake_enc_d[:, :, :-1]

        nodes_fake_pri, adj_fake_pri = self.model.generator(z_pri)
        nodes_fake_pri, adj_fake_pri = postprocess(nodes_fake_pri, "hard_gumbel"), postprocess(adj_fake_pri,
                                                                                               "hard_gumbel")
        adj_fake_pri = adj_fake_pri[:, :, :, 1]
        mask_fake_pri = nodes_fake_pri[:, :, -1] == 0
        nodes_fake_pri = nodes_fake_pri[:, :, :-1]

        d_logits_real, d_features_real = self.model.discriminator(nodes, adj, None)
        d_logits_fake_enc, d_features_fake_enc = self.model.discriminator(nodes_fake_enc_d, adj_fake_enc_d, None)
        d_logits_fake_pri, d_features_fake_pri = self.model.discriminator(nodes_fake_pri, adj_fake_pri, None)

        g_loss = kld_loss(mu, log_var)
        g_loss += self.criterion(d_logits_fake_enc, real_target) + self.criterion(d_logits_fake_pri, real_target)
        g_loss += mse_loss(d_features_fake_enc, d_features_real)

        g_loss.backward()
        self.opt_g.step()

        return d_loss, g_loss

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


def postprocess(logits, method, temperature=1.):
    shape = logits.shape
    if method == 'soft_gumbel':
        out = torch.nn.functional.gumbel_softmax(
            logits=logits.view(-1, shape[-1]) / temperature,
            hard=False
        )
    elif method == 'hard_gumbel':
        out = torch.nn.functional.gumbel_softmax(
            logits=logits.view(-1, shape[-1]) / temperature,
            hard=False
        )
    else:
        out = torch.nn.functional.softmax(
            input=logits / temperature
        )
    return out.view(shape)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    trainer = Trainer(args.__dict__)
    trainer.train()


