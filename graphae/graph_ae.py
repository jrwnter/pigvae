import torch
from graphae import encoder, decoder
from graphae.fully_connected import FNN


class Encoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.graph_encoder = encoder.GraphEncoder(
            input_dim=hparams["num_atom_features"],
            hidden_dim_gnn=hparams["graph_encoder_hidden_dim_gnn"],
            hidden_dim_fnn=hparams["graph_encoder_hidden_dim_fnn"],
            node_dim=hparams["node_dim"],
            num_nodes=hparams["max_num_nodes"],
            graph_emb_dim=hparams["graph_emb_dim"],
            perm_emb_dim=hparams["perm_emb_dim"],
            num_layers_gnn=hparams["graph_encoder_num_layers_gnn"],
            num_layers_fnn=hparams["graph_encoder_num_layers_fnn"],
            batch_norm=hparams["batch_norm"],
            non_linearity=hparams["nonlin"]
        )

    def forward(self, node_features, adj, mask=None):
        node_emb, perm_emb = self.graph_encoder(node_features, adj, mask)
        return node_emb, perm_emb


class Decoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fnn = FNN(
            input_dim=hparams["graph_emb_dim"] + hparams["perm_emb_dim"],
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

    def forward(self, graph_emb, perm_emb, do_postprocessing=False):
        x = torch.cat((graph_emb, perm_emb), dim=1)
        x = self.fnn(x)
        node_logits = self.node_predictor(x)
        adj_logits = self.edge_predictor(x)
        """if do_postprocessing:
            nodes = postprocess(
                logits=node_logits,
                method="hard_gumbel")
            adj = postprocess(
                logits=edge_logits,
                method="hard_gumbel")
            adj = adj[:, :, :, 1]"""
        return node_logits, adj_logits


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, node_features, adj, mask=None):
        graph_emb, perm_emb = self.encoder(node_features, adj, mask)
        node_features, adj = self.decoder(graph_emb, perm_emb)
        return node_features, adj

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


