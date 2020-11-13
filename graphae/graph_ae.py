import torch
from torch.nn import Linear, Parameter
from graphae import encoder, decoder


class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_nodes = hparams["max_num_nodes"]
        self.encoder = encoder.GraphEncoder(
            input_dim=hparams["num_node_features"],
            hidden_dim_gnn=hparams["graph_encoder_hidden_dim_gnn"],
            hidden_dim_fnn=hparams["graph_encoder_hidden_dim_fnn"],
            num_layers_gnn=hparams["graph_encoder_num_layers_gnn"],
            num_layers_fnn=hparams["graph_encoder_num_layers_fnn"],
            node_dim=hparams["node_dim"],
            num_nodes=hparams["max_num_nodes"],
            num_edge_features=hparams["num_edge_features"],
            batch_norm=hparams["batch_norm"],
            non_linearity=hparams["nonlin"],
            stack_node_emb=hparams["stack_node_emb"]
        )

    def forward(self, graph, noise=None):
        node_embs = self.encoder(graph)
        if noise is not None:
            node_embs = node_embs + noise * torch.randn_like(node_embs)
        return node_embs


class GraphDecoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.edge_predictor = decoder.EdgeDecoder(
            node_dim=hparams["node_dim"],
            hidden_dim=hparams["edge_decoder_hidden_dim"],
            num_layers=hparams["edge_decoder_num_layers"],
            num_edge_features=hparams["num_edge_features"],
            num_nodes=hparams["max_num_nodes"],
            non_lin=hparams["nonlin"],
            batch_norm=hparams["batch_norm"],
        )
        self.node_predictor = decoder.NodePredictor(
            node_dim=hparams["node_dim"],
            hidden_dim=hparams["node_decoder_hidden_dim"],
            num_layers=hparams["node_decoder_num_layers"],
            batch_norm=hparams["batch_norm"],
            num_node_features=hparams["num_node_features"],
            non_lin=hparams["nonlin"]
        )

    def forward(self, node_embs):
        node_logits = self.node_predictor(node_embs)
        adj_logits = self.edge_predictor(node_embs)
        mask_logits, node_logits = node_logits[:, :, 0], node_logits[:, :, 1:]
        return node_logits, adj_logits, mask_logits


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = GraphEncoder(hparams)
        self.decoder = GraphDecoder(hparams)
        self.node_dim = hparams["node_dim"]
        self.num_nodes = hparams["max_num_nodes"]
        self.empty_node = Parameter(torch.randn(self.node_dim))

    def encode(self, graph):
        node_embs = self.encoder(graph=graph, noise=None)
        return node_embs

    def forward(self, graph, postprocess_method=None, noise=None):
        node_embs = self.encode(graph=graph)
        node_logits, adj_logits, mask_logits = self.decoder(node_embs=node_embs)
        if postprocess_method is not None:
            node_logits, adj_logits = self.postprocess_logits(
                node_logits=node_logits,
                adj_logits=adj_logits,
                method=postprocess_method,
            )
        return node_logits, adj_logits, mask_logits

    def node_embs_to_dense(self, node_embs, batch_idxs):
        batch_size = batch_idxs.max().item() + 1
        device = node_embs.device
        mask = torch.where(
            torch.arange(self.num_nodes, device=device).unsqueeze(0) < torch.bincount(batch_idxs).unsqueeze(1),
            torch.ones(batch_size, self.num_nodes, device=device),
            torch.zeros(batch_size, self.num_nodes, device=device)
        ).bool().view(batch_size, self.num_nodes, 1).repeat(1, 1, self.node_dim)

        node_embs_dense = node_embs.new_zeros(batch_size, self.num_nodes, self.node_dim)
        node_embs_dense = node_embs_dense.masked_scatter_(mask, node_embs)

        return node_embs_dense, mask

    @staticmethod
    def postprocess_logits(node_logits, adj_logits, method=None, temp=1.0):
        element_type = node_logits[:, :, :11]
        charge_type = node_logits[:, :, 11:16]
        hybridization_type = node_logits[:, :, 16:]
        element_type = postprocess(element_type, method=method, temperature=temp)
        charge_type = postprocess(charge_type, method=method, temperature=temp)
        hybridization_type = postprocess(hybridization_type, method=method, temperature=temp)
        nodes = torch.cat((element_type, charge_type, hybridization_type), dim=-1)
        adj = postprocess(adj_logits, method=method)
        return nodes, adj

    @staticmethod
    def logits_to_one_hot(nodes, adj):
        batch_size, num_nodes = nodes.size(0), nodes.size(1)
        element_type = torch.argmax(nodes[:, :, :11], axis=-1).unsqueeze(-1)
        element_type = torch.zeros((batch_size, num_nodes, 11)).type_as(element_type).scatter_(2, element_type, 1)
        charge_type = torch.argmax(nodes[:, :, 11:16], axis=-1).unsqueeze(-1)
        charge_type = torch.zeros((batch_size, num_nodes, 5)).type_as(charge_type).scatter_(2, charge_type, 1)
        hybridization_type = torch.argmax(nodes[:, :, 16:], axis=-1).unsqueeze(-1)
        hybridization_type = torch.zeros((batch_size, num_nodes, 7)).type_as(hybridization_type).scatter_(2, hybridization_type, 1)
        nodes = torch.cat((element_type, charge_type, hybridization_type), dim=-1)
        adj_shape = adj.shape
        adj = torch.argmax(adj, axis=-1).unsqueeze(-1)
        adj = torch.zeros(adj_shape).type_as(adj).scatter_(3, adj, 1)
        return nodes, adj

def postprocess(logits, method, temperature=1.):
    if method == 'soft_gumbel':
        out = torch.nn.functional.gumbel_softmax(
            logits=logits,
            hard=False,
            tau=temperature,
            dim=-1
        )
    elif method == 'hard_gumbel':
        out = torch.nn.functional.gumbel_softmax(
            logits=logits,
            hard=True,
            tau=temperature,
            dim=-1
        )
    elif method == "softmax":
        out = torch.nn.functional.softmax(
            input=logits,
            dim=-1
        )
    else:
        raise NotImplementedError
    return out



