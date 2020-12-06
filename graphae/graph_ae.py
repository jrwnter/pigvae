import torch
from torch.nn import Linear, LayerNorm, Dropout
#from graphae import encoder, decoder
from graphae.graph_transformer import GraphTransformer, PositionalEncoding
from graphae import permuter
from graphae.data import DenseGraphBatch


class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.graph_transformer = GraphTransformer(
            node_hidden_dim=512,
            edge_hidden_dim=512,
            nk_dim=64,
            ek_dim=64,
            v_dim=64,
            num_heads=8,
            ppf_hidden_dim=1024,
            num_layers=4,
        )
        self.node_fc_in = Linear(hparams["num_node_features"] + 1, 512)
        self.edge_fc_in = Linear(hparams["num_edge_features"] + 1, 512)
        self.dropout = Dropout(0.1)

    #  TODO: only apply fc on masked features. check speed up
    def forward(self, node_features, edge_features, mask):

        node_features = self.dropout(self.node_fc_in(node_features))
        edge_features = self.dropout(self.edge_fc_in(edge_features))
        node_features, edge_features = self.graph_transformer(node_features, edge_features, mask)
        return node_features, edge_features


class GraphDecoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.posiotional_embedding = PositionalEncoding(512)
        self.graph_transformer = self.encoder = GraphTransformer(
            node_hidden_dim=512,
            edge_hidden_dim=512,
            nk_dim=64,
            ek_dim=64,
            v_dim=64,
            num_heads=8,
            ppf_hidden_dim=1024,
            num_layers=4
        )
        self.node_fc_in = Linear(512, 512)
        self.edge_fc_in = Linear(2 * 512, 512)
        self.node_fc_out = Linear(512, hparams["num_node_features"])
        self.edge_fc_out = Linear(512, hparams["num_edge_features"] + 1)
        self.dropout = Dropout(0.1)

    def forward(self, graph_emb, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        node_features = graph_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        node_features = node_features + self.posiotional_embedding(batch_size, num_nodes)
        node_features = torch.matmul(perm, node_features)
        node_features_combined = torch.cat(
            (node_features.unsqueeze(1).repeat(1, num_nodes, 1, 1),
             node_features.unsqueeze(2).repeat_interleave(num_nodes, dim=2)),
            dim=-1)  # b x nn x nn x 2*512

        node_features = self.dropout(self.node_fc_in(node_features))
        edge_features = self.dropout(self.edge_fc_in(node_features_combined))
        node_features, edge_features = self.graph_transformer(node_features, edge_features, mask)
        node_features = self.node_fc_out(node_features)
        edge_features = self.edge_fc_out(edge_features)

        return node_features, edge_features


class Permuter(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.permuter = permuter.Permuter(
            input_dim=512,
        )

    def forward(self, node_features, mask, hard=False, tau=1.0):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        perm = self.permuter(node_features, mask, hard=hard, tau=tau)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = GraphEncoder(hparams)
        self.permuter = Permuter(hparams)
        self.decoder = GraphDecoder(hparams)
        self.node_dim = hparams["node_dim"]
        self.num_nodes = hparams["max_num_nodes"]

    def encode(self, graph):
        node_embs, _ = self.encoder(
            node_features=graph.node_features,
            edge_features=graph.edge_features,
            mask=graph.mask
        )
        return node_embs

    def forward(self, graph, training, tau):
        node_embs = self.encode(graph=graph)
        graph_emb, node_embs = node_embs[:, 0], node_embs[:, 1:]
        perm = self.permuter(node_embs, mask=graph.mask[:, 1:], hard=not training, tau=tau)
        node_logits, edge_logits = self.decoder(
            graph_emb=graph_emb,
            perm=perm,
            mask=graph.mask[:, 1:]
        )
        graph_pred = DenseGraphBatch(
            node_features=node_logits,
            edge_features=edge_logits,
            mask=graph.mask[:, 1:],
            molecular_properties=None
        )
        return graph_pred, graph_emb, perm

    @staticmethod
    def logits_to_one_hot(graph):
        nodes = graph.node_features[:, :, :-1]
        edges = graph.edge_features[:, :, :]
        batch_size, num_nodes = nodes.size(0), nodes.size(1)
        element_type = torch.argmax(nodes[:, :, :11], axis=-1).unsqueeze(-1)
        element_type = torch.zeros((batch_size, num_nodes, 11)).type_as(element_type).scatter_(2, element_type, 1)
        charge_type = torch.argmax(nodes[:, :, 11:16], axis=-1).unsqueeze(-1)
        charge_type = torch.zeros((batch_size, num_nodes, 5)).type_as(charge_type).scatter_(2, charge_type, 1)
        num_explicit_hydrogens = torch.argmax(nodes[:, :, 16:], axis=-1).unsqueeze(-1)
        num_explicit_hydrogens = torch.zeros((batch_size, num_nodes, 4)).type_as(num_explicit_hydrogens).scatter_(2, num_explicit_hydrogens, 1)
        nodes = torch.cat((element_type, charge_type, num_explicit_hydrogens), dim=-1)
        edges_shape = edges.shape
        edges = torch.argmax(edges, axis=-1).unsqueeze(-1)
        edges = torch.zeros(edges_shape).type_as(edges).scatter_(3, edges, 1)
        graph.node_features = nodes
        graph.edge_features = edges
        return graph


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



