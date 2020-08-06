import numpy as np
import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from graphae.graph_ae import GraphAE, Encoder, Descriminator, Decoder
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

        self.generator = Decoder(hparams).to(self.device)
        self.descriminator = Descriminator(hparams).to(self.device)

        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=0.00005, betas=(0.5, 0.99))
        self.opt_disc = torch.optim.Adam(self.descriminator.parameters(), lr=0.00005, betas=(0.5, 0.99))
        self.global_step = 0
        self.num_epochs = 10000

    def train(self):
        log = {"gen_loss": [], "disc_loss": []}
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            for batch in self.train_dataloader:
                self.global_step += 1
                if self.global_step % 5 == 0:
                    disc_loss, gen_loss = self.train_step(batch, train_generator=True)
                    log["disc_loss"].append(disc_loss.item())
                    log["gen_loss"].append(gen_loss.item())
                else:
                    disc_loss = self.train_step(batch, train_generator=False)
                    log["disc_loss"].append(disc_loss.item())
                if self.global_step % 100 == 0:
                    self.evaluate(train_log=log)
                    torch.save(self.generator, self.save_file)
                    log = {"gen_loss": [], "disc_loss": []}

    def train_step(self, batch, train_generator):
        nodes, adj, mask = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        # Disciminator

        self.opt_disc.zero_grad()
        z = torch.randn([adj.shape[0], 128]).to(self.device)
        nodes_pred, adj_pred = self.generator(z)
        nodes_pred, adj_pred = postprocess(nodes_pred, "hard_gumbel"), postprocess(adj_pred, "hard_gumbel")
        adj_pred = adj_pred[:, :, :, 1]
        mask_pred = nodes_pred[:, :, -1] == 0
        nodes_pred = nodes_pred[:, :, :-1]

        fake_pred = self.descriminator(nodes_pred.detach(), adj_pred.detach(), None)
        real_pred = self.descriminator(nodes, adj, None)

        # Compute loss for gradient penalty.
        eps = torch.rand(real_pred.size(0), 1, 1).to(self.device)
        x_int0 = (eps * nodes + (1. - eps) * nodes_pred).requires_grad_(True)
        x_int1 = (eps * adj + (1. - eps) * adj_pred).requires_grad_(True)
        grad = self.descriminator(x_int0, x_int1, None)
        d_loss_gp = self.gradient_penalty(grad, x_int0) + self.gradient_penalty(grad, x_int1)

        disc_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + 10 * d_loss_gp

        disc_loss.backward()
        self.opt_disc.step()

        # Generator
        self.opt_gen.zero_grad()

        if train_generator:
            nodes_pred, adj_pred = self.generator(z)
            nodes_pred, adj_pred = postprocess(nodes_pred, "hard_gumbel"), postprocess(adj_pred, "hard_gumbel")
            adj_pred = adj_pred[:, :, :, 1]
            mask_pred = nodes_pred[:, :, -1] == 0
            nodes_pred = nodes_pred[:, :, :-1]

            fake_pred = self.descriminator(nodes_pred, adj_pred, None)
            gen_loss = - torch.mean(fake_pred)

            gen_loss.backward()
            self.opt_gen.step()
            return disc_loss, gen_loss
        return disc_loss

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


