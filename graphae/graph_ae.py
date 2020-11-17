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

    def forward(self, graph):
        node_embs = self.encoder(graph)
        return node_embs


class GraphDecoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.node_emb_decoder = decoder.NodeEmbDecoder(
            input_dim=hparams["node_dim"],
            hidden_dim=hparams["node_emb_decoder_hidden_dim"],
            num_layers=hparams["node_emb_decoder_num_layers"],
            output_dim=hparams["node_emb_decoder_hidden_dim"],
            num_heads=4,
            non_lin=hparams["nonlin"],
            batch_norm=hparams["batch_norm"],
        )

        self.edge_predictor = decoder.EdgeTypePredictor(
            input_dim=hparams["node_emb_decoder_hidden_dim"],
            hidden_dim=hparams["edge_decoder_hidden_dim"],
            output_dim=hparams["num_edge_features"] + 1,
            num_layers=hparams["edge_decoder_num_layers"],
            non_lin=hparams["nonlin"],
            batch_norm=hparams["batch_norm"],
        )
        self.node_predictor = decoder.NodeTypePredictor(
            input_dim=hparams["node_emb_decoder_hidden_dim"],
            hidden_dim=hparams["node_decoder_hidden_dim"],
            output_dim=hparams["num_node_features"],
            num_layers=hparams["node_decoder_num_layers"],
            non_lin=hparams["nonlin"],
            batch_norm=hparams["batch_norm"],
        )

    def forward(self, x, edge_index):
        x = self.node_emb_decoder(x, edge_index)
        node_logits = self.node_predictor(x)
        edge_logits = self.edge_predictor(x, edge_index)
        return node_logits, edge_logits


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = GraphEncoder(hparams)
        self.decoder = GraphDecoder(hparams)
        self.node_dim = hparams["node_dim"]
        self.num_nodes = hparams["max_num_nodes"]

    def encode(self, graph):
        node_embs = self.encoder(graph=graph)
        return node_embs

    def forward(self, graph, postprocess_method=None):
        node_embs = self.encode(graph=graph)
        node_logits, edge_logits = self.decoder(
            x=node_embs,
            edge_index=graph.dense_edge_index,
        )
        if postprocess_method is not None:
            node_logits, edge_logits = self.postprocess_logits(
                node_logits=node_logits,
                edge_logits=edge_logits,
                method=postprocess_method,
            )
        return node_logits, edge_logits

    @staticmethod
    def postprocess_logits(node_logits, edge_logits, method=None, temp=1.0):
        element_type = node_logits[:, :11]
        charge_type = node_logits[:, 11:16]
        hybridization_type = node_logits[:, 16:]
        element_type = postprocess(element_type, method=method, temperature=temp)
        charge_type = postprocess(charge_type, method=method, temperature=temp)
        hybridization_type = postprocess(hybridization_type, method=method, temperature=temp)
        nodes = torch.cat((element_type, charge_type, hybridization_type), dim=-1)
        edges = postprocess(edge_logits, method=method)
        return nodes, edges

    @staticmethod
    def logits_to_one_hot(nodes, edges):
        batch_size, num_nodes = nodes.size(0), nodes.size(1)
        element_type = torch.argmax(nodes[:, :, :11], axis=-1).unsqueeze(-1)
        element_type = torch.zeros((batch_size, num_nodes, 11)).type_as(element_type).scatter_(2, element_type, 1)
        charge_type = torch.argmax(nodes[:, :, 11:16], axis=-1).unsqueeze(-1)
        charge_type = torch.zeros((batch_size, num_nodes, 5)).type_as(charge_type).scatter_(2, charge_type, 1)
        hybridization_type = torch.argmax(nodes[:, :, 16:], axis=-1).unsqueeze(-1)
        hybridization_type = torch.zeros((batch_size, num_nodes, 7)).type_as(hybridization_type).scatter_(2, hybridization_type, 1)
        nodes = torch.cat((element_type, charge_type, hybridization_type), dim=-1)
        edges_shape = edges.shape
        edges = torch.argmax(edges, axis=-1).unsqueeze(-1)
        edges = torch.zeros(edges_shape).type_as(edges).scatter_(1, edges, 1)
        return nodes, edges


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



