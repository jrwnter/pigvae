import pytorch_lightning as pl
from graphae.graph_ae import GraphAE
from graphae.metrics import *
from pivae.vae import PIVAE
from graphae.data import get_mask_for_batch
from graphae.side_tasks import PropertyPredictor
from graphae.ddp import MyDistributedDataParallel


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
            step_size=1
        )

    def forward(self, graph, training, tau, postprocess_method=None):
        postprocess_method = self.get_postprocess_method(postprocess_method)
        mask = get_mask_for_batch(batch=graph.batch, device=graph.x.device)
        node_embs = self.graph_ae.encode(graph=graph)
        node_embs_pred, perm, graph_emb, mu, logvar = self.pi_ae(
            x=node_embs,
            mask=mask,
            batch=graph.batch,
            training=training,
            tau=tau
        )
        node_logits, adj_logits = self.graph_ae.decoder(
            x=node_embs_pred,
            edge_index=graph.dense_edge_index,
            batch=graph.batch
        )
        props_pred = self.property_predictor(graph_emb)
        if postprocess_method is not None:
            node_logits, adj_logits = self.postprocess_logits(
                node_logits=node_logits,
                adj_logits=adj_logits,
                method=postprocess_method,
            )
        return node_logits, adj_logits, perm, graph_emb, props_pred, mu, logvar

    def training_step(self, graph, batch_idx):
        nodes_true, edges_true, props_true = graph.x, graph.dense_edge_attr, graph.mol_properties
        tau = self.tau_scheduler.tau
        nodes_pred, edges_pred, perm, graph_emb, props_pred, mu, logvar = self(
            graph=graph,
            training=True,
            tau=tau
        )
        loss = self.critic(
            nodes_true=nodes_true,
            edges_true=edges_true,
            nodes_pred=nodes_pred,
            edges_pred=edges_pred,
            perm=perm,
            props_true=props_true,
            props_pred=props_pred,
            mu=mu,
            logvar=logvar
        )
        self.log_dict(loss)
        self.log("tau", tau)
        return loss

    def validation_step(self, graph, batch_idx):
        nodes_true, edges_true, props_true = graph.x, graph.dense_edge_attr, graph.mol_properties
        tau = self.tau_scheduler.tau
        nodes_pred, edges_pred, perm, graph_emb, props_pred, mu, logvar = self(
            graph=graph,
            training=True,
            tau=tau
        )
        metrics_soft = self.critic.evaluate(
            nodes_true=nodes_true,
            edges_true=edges_true,
            nodes_pred=nodes_pred,
            edges_pred=edges_pred,
            perm=perm,
            props_true=props_true,
            props_pred=props_pred,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        nodes_pred, edges_pred, perm, graph_emb, props_pred, mu, logvar  = self(
            graph=graph,
            training=False,
            tau=tau
        )
        metrics_hard = self.critic.evaluate(
            nodes_true=nodes_true,
            edges_true=edges_true,
            nodes_pred=nodes_pred,
            edges_pred=edges_pred,
            perm=perm,
            props_true=props_true,
            props_pred=props_pred,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics)

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
            'monitor': 'val_hard_loss',
            'frequency': self.hparams["eval_freq"] + 1
        }
        return [optimizer], [scheduler]

    def get_postprocess_method(self, postprocess_method):
        if postprocess_method is None:
            if self.hparams["postprocess_method"] == 0:
                postprocess_method = None
            elif self.hparams["postprocess_method"] == 1:
                postprocess_method = "soft_gumbel"
            elif self.hparams["postprocess_method"] == 2:
                postprocess_method = "hard_gumbel"
            elif self.hparams["postprocess_method"] == 3:
                postprocess_method = "softmax"
            else:

                raise NotImplementedError
        return postprocess_method


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
