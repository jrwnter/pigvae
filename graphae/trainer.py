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
        graph_pred, perm, mu, logvar = self.graph_ae(graph, training, tau)
        return graph_pred, perm, mu, logvar

    def training_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
            tau=self.hparams["tau"]
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )
        self.log_dict(loss)
        return loss

    def validation_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
            tau=self.hparams["tau"]
        )
        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=False,
            tau=self.hparams["tau"]
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics)
        self.log_dict(metrics_soft)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.98))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: 0.03 * np.minimum(np.power(step, -0.5), step * np.power(4000, -1.5))
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'frequency': self.hparams["eval_freq"] + 1
        }
        return [optimizer], [scheduler]
