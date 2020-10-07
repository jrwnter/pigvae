import torch
from graphae import encoder, decoder, permuter, sinkhorn_ops
from graphae.fully_connected import FNN


class Encoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.stack_node_emb = hparams["stack_node_emb"]

        self.graph_encoder = encoder.GraphEncoder(
            input_dim=hparams["num_atom_features"],
            hidden_dim=hparams["graph_encoder_hidden_dim_gnn"],
            node_dim=hparams["node_dim"],
            num_nodes=hparams["max_num_nodes"],
            num_layers=hparams["graph_encoder_num_layers_gnn"],
            batch_norm=hparams["batch_norm"],
            non_linearity=hparams["nonlin"],
            stack_node_emb=hparams["stack_node_emb"]
        )
        self.node_aggregator = encoder.NodeAggregator(
            input_dim=hparams["graph_encoder_num_layers_gnn"] * hparams["node_dim"] if hparams["stack_node_emb"] else hparams["node_dim"],
            emb_dim=hparams["graph_emb_dim"],
            hidden_dim=hparams["graph_encoder_hidden_dim_fnn"],
            num_layers=hparams["graph_encoder_num_layers_fnn"],
            batch_norm=hparams["batch_norm"],
            non_linearity=hparams["nonlin"]
        )

    def forward(self, node_features, adj, mask=None):
        node_embs = self.graph_encoder(node_features, adj, mask)
        graph_emb = self.node_aggregator(node_embs)
        if self.stack_node_emb:
            node_embs = node_embs[:, :, -1]  # just take output (node emb) of the last layer
        return graph_emb, node_embs


class Decoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fnn = FNN(
            input_dim=hparams["graph_emb_dim"],
            hidden_dim=hparams["meta_node_decoder_hidden_dim"],
            output_dim=hparams["meta_node_decoder_hidden_dim"],
            num_layers=hparams["meta_node_decoder_num_layers"],
            non_linearity=hparams["nonlin"],
            batch_norm=hparams["batch_norm"],
        )
        self.edge_predictor = decoder.EdgePredictor(
            num_nodes=hparams["max_num_nodes"],
            input_dim=hparams["meta_node_decoder_hidden_dim"],
            hidden_dim=hparams["edge_predictor_hidden_dim"],
            num_layers=hparams["edge_predictor_num_layers"],
            batch_norm=hparams["batch_norm"],
            non_lin=hparams["nonlin"]
        )
        self.node_predictor = decoder.NodePredictor(
            num_nodes=hparams["max_num_nodes"],
            input_dim=hparams["meta_node_decoder_hidden_dim"],
            hidden_dim=hparams["node_decoder_hidden_dim"],
            num_layers=hparams["node_decoder_num_layers"],
            batch_norm=hparams["batch_norm"],
            num_node_features=hparams["num_atom_features"],
            non_lin=hparams["nonlin"]
        )

    def forward(self, graph_emb):
        x = self.fnn(graph_emb)
        node_logits = self.node_predictor(x)
        adj_logits = self.edge_predictor(x)
        return node_logits, adj_logits


class Permuter(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.permuter = permuter.SinkhornNetwork(
            input_dim=hparams["node_dim"],
            hidden_dim=hparams["permuter_hidden_dim"],
            num_layers=hparams["permuter_num_layers"],
            num_nodes=hparams["max_num_nodes"],
            batch_norm=hparams["batch_norm"],
            non_linearity=hparams["nonlin"]
        )

    def forward(self, node_embs, training=True):
        p_log_alpha = self.permuter(node_embs)
        # apply the gumbel sinkhorn on log alpha
        perms, log_alpha_w_noise = sinkhorn_ops.my_gumbel_sinkhorn(
            log_alpha=p_log_alpha,
            temp=self.hparams["sinkhorn_temp"],
            n_samples=self.hparams["samples_per_graph"],
            noise_factor=self.hparams["sinkhorn_noise_factor"] if training else 0.0,
            n_iters=self.hparams["sinkhorn_num_iterations"],
            squeeze=True)
        """print(soft_perms_inf.shape)
        inv_soft_perms_flat = sinkhorn_ops.inv_soft_pers_flattened(
            soft_perms_inf=soft_perms_inf,
            n_numbers=self.hparams["max_num_nodes"]
        )"""
        perms = torch.transpose(perms, 1, 2)
        return perms


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.permuter = Permuter(hparams)

    def forward(self, node_features, adj, mask=None, training=True):
        graph_emb, node_embs = self.encoder(node_features, adj, mask)
        node_features, adj = self.decoder(graph_emb)
        perms = self.permuter(node_embs, training=training)
        node_features = torch.matmul(perms, node_features)
        shape = adj.shape
        adj = torch.matmul(perms, adj.view(shape[0], shape[1], shape[2] * shape[3])).view(shape)
        return node_features, adj, perms

    @staticmethod
    def logits_to_one_hot(nodes, adj):
        nodes_shape = nodes.shape
        nodes = torch.argmax(nodes, axis=-1).unsqueeze(-1)
        nodes = torch.zeros(nodes_shape).type_as(nodes).scatter_(2, nodes, 1)
        #nodes = nodes[:, :, :-1]  # remove empty node
        adj_shape = adj.shape
        adj = torch.argmax(adj, axis=-1).unsqueeze(-1)
        adj = torch.zeros(adj_shape).type_as(adj).scatter_(3, adj, 1)
        return nodes, adj


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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
            hard=True
        )
    else:
        out = torch.nn.functional.softmax(
            input=logits / temperature
        )
    return out.view(shape)


