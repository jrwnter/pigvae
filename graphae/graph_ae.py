import torch
from torch.nn import Linear, LayerNorm, Dropout
from torch.nn.functional import relu, pad
from graphae.graph_transformer import Transformer, PositionalEncoding
from graphae import permuter
from graphae.data import DenseGraphBatch
from graphae.side_tasks import PropertyPredictor


MEAN_DISTANCE = 2.0626
STD_DISTANCE = 1.1746



class GraphEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.posiotional_embedding = PositionalEncoding(32)
        self.graph_transformer = Transformer(
            hidden_dim=hparams["graph_encoder_hidden_dim"],
            k_dim=hparams["graph_encoder_k_dim"],
            v_dim=hparams["graph_encoder_v_dim"],
            num_heads=hparams["graph_encoder_num_heads"],
            ppf_hidden_dim=hparams["graph_encoder_ppf_hidden_dim"],
            num_layers=hparams["graph_encoder_num_layers"],
        )
        # 11 edge features (including empty edge) and 26 node features + emb node feature and emb node edge
        self.fc_in = Linear(1 * 2 + 1, hparams["graph_encoder_hidden_dim"])
        self.layer_norm = LayerNorm(hparams["graph_encoder_hidden_dim"])
        self.dropout = Dropout(0.1)

    def add_emb_node_and_feature(self, node_features, edge_features, mask):
        node_features = pad(node_features, (0, 0, 1, 0))
        edge_features = pad(edge_features, (0, 0, 1, 0, 1, 0))
        mask = pad(mask, (1, 0), value=1)
        return node_features, edge_features, mask

    def forward(self, node_features, edge_features, mask):
        node_features, edge_features, mask = self.add_emb_node_and_feature(node_features, edge_features, mask)
        batch_size, num_nodes = node_features.size(0), node_features.size(1)

        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        node_features_combined = torch.cat(
            (node_features.unsqueeze(2).repeat(1, 1, num_nodes, 1),
             node_features.unsqueeze(1).repeat_interleave(num_nodes, dim=1)),
            dim=-1)
        x = torch.cat((edge_features, node_features_combined), dim=-1)
        #print(x.shape, edge_features.shape, node_features.shape, node_features_combined.shape)
        x = self.layer_norm(self.dropout(self.fc_in(x)))
        x = self.graph_transformer(x, mask=edge_mask)
        node_features = torch.diagonal(x, dim1=1, dim2=2).transpose(1, 2)
        graph_emb, node_features = node_features[:, 0], node_features[:, 1:]
        return graph_emb, node_features


class GraphDecoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.posiotional_embedding = PositionalEncoding(hparams["graph_decoder_pos_emb_dim"])
        self.graph_transformer = Transformer(
            hidden_dim=hparams["graph_decoder_hidden_dim"],
            k_dim=hparams["graph_decoder_k_dim"],
            v_dim=hparams["graph_decoder_v_dim"],
            num_heads=hparams["graph_decoder_num_heads"],
            ppf_hidden_dim=hparams["graph_decoder_ppf_hidden_dim"],
            num_layers=hparams["graph_decoder_num_layers"],
        )
        self.fc_in = Linear(hparams["graph_decoder_hidden_dim"] + 2 * hparams["graph_decoder_pos_emb_dim"],
                            hparams["graph_decoder_hidden_dim"])
        #self.node_fc_out = Linear(hparams["graph_decoder_hidden_dim"], 20)
        self.edge_fc_out = Linear(hparams["graph_decoder_hidden_dim"], 1)
        self.dropout = Dropout(0.1)
        self.layer_norm = LayerNorm(hparams["graph_decoder_hidden_dim"])

    def forward(self, graph_emb, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        pos_emb = self.posiotional_embedding(batch_size, num_nodes)
        if perm is not None:
            pos_emb = torch.matmul(perm, pos_emb)
        pos_emb_combined = torch.cat(
            (pos_emb.unsqueeze(2).repeat(1, 1, num_nodes, 1),
             pos_emb.unsqueeze(1).repeat_interleave(num_nodes, dim=1)),
            dim=-1)

        x = graph_emb.unsqueeze(1).unsqueeze(1).expand(-1, num_nodes, num_nodes, -1)
        x = torch.cat((x, pos_emb_combined), dim=-1)
        x = self.layer_norm(self.dropout(self.fc_in(x)))
        x = self.graph_transformer(x, mask=edge_mask)

        node_features = torch.diagonal(x, dim1=1, dim2=2).transpose(1, 2)
        edge_features = x

        edge_features = self.edge_fc_out(edge_features)
        self_edge_mask = torch.eye(num_nodes, num_nodes, device=node_features.device).bool().unsqueeze(-1)
        edge_features.masked_fill_(self_edge_mask, 0)
        edge_features = (edge_features + edge_features.permute(0, 2, 1, 3)) / 2

        return edge_features


class Permuter(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.permuter = permuter.Permuter(
            input_dim=hparams["graph_decoder_hidden_dim"],
        )

    def forward(self, node_features, mask, hard=False, tau=1.0):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        node_features = node_features + torch.randn_like(node_features) * 0.05
        perm = self.permuter(node_features, mask, hard=hard, tau=tau)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    @staticmethod
    def permute_node_features(node_features, perm):
        node_features = torch.matmul(perm, node_features)
        return node_features

    @staticmethod
    def permute_edge_features(edge_features, perm):
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features)
        edge_features = torch.matmul(perm.unsqueeze(1), edge_features.permute(0, 2, 1, 3))
        edge_features = edge_features.permute(0, 2, 1, 3)
        return edge_features

    @staticmethod
    def permute_graph(graph, perm):
        graph.node_features = Permuter.permute_node_features(graph.node_features, perm)
        graph.edge_features = Permuter.permute_edge_features(graph.edge_features, perm)
        return graph


class BottleNeckEncoder(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = Linear(d_in, d_out)

    def forward(self, x):
        x = self.w(relu(x))
        return x


class BottleNeckDecoder(torch.nn.Module):
    def __init__(self, d_in,  d_out):
        super().__init__()
        self.w = Linear(d_in, d_out)

    def forward(self, x):
        x = self.w(x)
        return x

# TODO: get attn mask for encoder and decoder together
class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = GraphEncoder(hparams)
        self.bottle_neck_encoder = BottleNeckEncoder(hparams["graph_encoder_hidden_dim"], hparams["emb_dim"])
        self.bottle_neck_decoder = BottleNeckDecoder(hparams["emb_dim"], hparams["graph_decoder_hidden_dim"])
        self.num_atoms_predictor = PropertyPredictor(hparams["emb_dim"], 256, 1)
        self.permuter = Permuter(hparams)
        self.decoder = GraphDecoder(hparams)

    def encode(self, graph):
        node_features = graph.node_features
        edge_features = graph.edge_features
        mask = graph.mask

        graph_emb, node_features = self.encoder(
            node_features=node_features,
            edge_features=edge_features,
            mask=mask,
        )
        graph_emb = self.bottle_neck_encoder(graph_emb)
        return graph_emb, node_features

    def decode(self, graph_emb, perm, mask):
        num_atoms = self.num_atoms_predictor(graph_emb).squeeze()
        graph_emb = self.bottle_neck_decoder(graph_emb)
        edge_logits = self.decoder(
            graph_emb=graph_emb,
            perm=perm,
            mask=mask
        )
        graph_pred = DenseGraphBatch(
            node_features=mask,
            edge_features=edge_logits,
            mask=mask,
            num_atoms=num_atoms
        )
        return graph_pred

    def forward(self, graph, training, tau):
        graph_emb, node_features = self.encode(graph=graph)
        perm = self.permuter(node_features, mask=graph.mask, hard=not training, tau=tau)
        graph_pred = self.decode(graph_emb, perm, graph.mask)
        return graph_pred, perm

    @staticmethod
    def logits_to_one_hot(graph):
        nodes = graph.node_features
        edges = graph.edge_features
        batch_size, num_nodes = nodes.size(0), nodes.size(1)
        element_type = torch.argmax(nodes[:, :, :11], axis=-1).unsqueeze(-1)
        element_type = torch.zeros((batch_size, num_nodes, 11)).type_as(element_type).scatter_(2, element_type, 1)
        charge_type = torch.argmax(nodes[:, :, 11:16], axis=-1).unsqueeze(-1)
        charge_type = torch.zeros((batch_size, num_nodes, 5)).type_as(charge_type).scatter_(2, charge_type, 1)
        num_explicit_hydrogens = torch.argmax(nodes[:, :, 16:], axis=-1).unsqueeze(-1)
        num_explicit_hydrogens = torch.zeros((batch_size, num_nodes, 4)).type_as(num_explicit_hydrogens).scatter_(
            2, num_explicit_hydrogens, 1)
        nodes = torch.cat((element_type, charge_type, num_explicit_hydrogens), dim=-1)
        edges_shape = edges.shape
        edges = torch.argmax(edges, axis=-1).unsqueeze(-1)
        edges = torch.zeros(edges_shape).type_as(edges).scatter_(3, edges, 1)
        graph.node_features = nodes
        graph.edge_features = edges
        return graph
