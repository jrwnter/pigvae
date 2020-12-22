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
        self.hparams = hparams
        self.graph_ae = GraphAE(hparams)
        #self.pi_ae = PIVAE(hparams)
        self.critic = Critic(hparams["alpha"])
        """self.tau_scheduler = TauScheduler(
            start_value=hparams["tau"],
            factor=0.98,
            step_size=1
        )"""

    def forward(self, graph, training, tau):
        graph_pred, graph_emb, perm = self.graph_ae(graph, training, tau)
        return graph_pred, graph_emb, perm

    def training_step(self, graph, batch_idx):
        graph_pred, graph_emb, perm = self(
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
        graph_pred, graph_emb, perm = self(
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
        graph_pred, graph_emb, perm = self(
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
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.75,
            patience=5,
            cooldown=20,
            min_lr=1e-6,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'frequency': self.hparams["eval_freq"] + 1
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 10000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 10000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
