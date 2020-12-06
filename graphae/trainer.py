import pytorch_lightning as pl
from graphae.graph_ae import GraphAE
from graphae.metrics import *
from pivae.vae import PIVAE
from graphae.side_tasks import PropertyPredictor


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        hparams["max_num_elements"] = hparams["max_num_nodes"]
        hparams["element_dim"] = hparams["node_dim"]
        hparams["element_emb_dim"] = hparams["node_dim"]
        if "tau" not in hparams:
            hparams["tau"] = 1.0
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)
        self.pi_ae = PIVAE(hparams)
        self.property_predictor = PropertyPredictor(hparams)
        self.critic = Critic(hparams["alpha"])
        self.tau_scheduler = TauScheduler(
            start_value=hparams["tau"],
            factor=0.95,
            step_size=5
        )

    def forward(self, graph, training, tau):
        graph_pred, graph_emb, perm = self.graph_ae(graph, training, tau)
        return graph_pred, graph_emb, perm

    def training_step(self, graph, batch_idx):
        tau = self.tau_scheduler.tau
        graph_pred, perm, graph_emb = self(
            graph=graph,
            training=True,
            tau=tau
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
        )
        self.log_dict(loss)
        self.log("tau", tau)
        return loss

    def validation_step(self, graph, batch_idx):
        tau = self.tau_scheduler.tau
        graph_pred, perm, graph_emb = self(
            graph=graph,
            training=True,
            tau=tau
        )
        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            prefix="val",
        )
        """graph_pred, perm, graph_emb = self(
            graph=graph,
            training=False,
            tau=tau
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            prefix="val_hard",
        )
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics)"""
        self.log_dict(metrics_soft)

    def on_validation_epoch_end(self):
        self.tau_scheduler()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=20,
            cooldown=50,
            min_lr=1e-6,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'frequency': self.hparams["eval_freq"] + 1
        }
        return [optimizer], [scheduler]


class TauScheduler(object):
    def __init__(self, start_value, factor, step_size):
        self.tau = start_value
        self.factor = factor
        self.step_size = step_size
        self.steps = 0

    def __call__(self):
        self.steps += 1
        if self.steps >= self.step_size:
            self.tau *= self.factor
            self.steps = 0
