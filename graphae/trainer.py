import pytorch_lightning as pl
from graphae.graph_ae import GraphAE
from graphae.metrics import *


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)
        self.critic = Critic(hparams["alpha"])

    def forward(self, graph, training, tau):
        graph_pred, perm = self.graph_ae(graph, training, tau)
        return graph_pred, perm

    def training_step(self, graph, batch_idx):
        graph_pred, perm = self(
            graph=graph,
            training=True,
            tau=self.hparams["tau"]
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
        )
        self.log_dict(loss)
        return loss

    def validation_step(self, graph, batch_idx):
        graph_pred, perm = self(
            graph=graph,
            training=True,
            tau=self.hparams["tau"]
        )
        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            prefix="val",
        )
        graph_pred, perm = self(
            graph=graph,
            training=False,
            tau=self.hparams["tau"]
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            prefix="val_hard",
        )
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics)
        self.log_dict(metrics_soft)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.98))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.995,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.hparams["eval_freq"] + 1
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 5000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 5000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
