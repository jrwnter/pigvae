import torch
from torch.nn import Linear, LayerNorm, Dropout
from torch.nn.functional import relu, pad
from pigvae.graph_transformer import Transformer, PositionalEncoding
from pigvae.synthetic_graphs.data import DenseGraphBatch


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.vae = hparams["vae"]
        self.encoder = GraphEncoder(hparams)
        self.bottle_neck_encoder = BottleNeckEncoder(hparams)
        self.bottle_neck_decoder = BottleNeckDecoder(hparams)
        self.property_predictor = PropertyPredictor(hparams)
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
        graph_emb, mu, logvar = self.bottle_neck_encoder(graph_emb)
        return graph_emb, node_features, mu, logvar

    def decode(self, graph_emb, perm, mask=None):
        props = self.property_predictor(graph_emb).squeeze()
        if mask is None:
            num_nodes = torch.round(props * STD_NUM_NODES + MEAN_NUM_NODES).long()
            mask = torch.arange(max(num_nodes)).type_as(num_nodes).unsqueeze(0) < num_nodes.unsqueeze(1)
        graph_emb = self.bottle_neck_decoder(graph_emb)
        node_logits, edge_logits = self.decoder(
            graph_emb=graph_emb,
            perm=perm,
            mask=mask
        )
        graph_pred = DenseGraphBatch(
            node_features=node_logits,
            edge_features=edge_logits,
            mask=mask,
            properties=props
        )
        return graph_pred

    def forward(self, graph, training, tau):
        graph_emb, node_features, mu, logvar = self.encode(graph=graph)
        perm = self.permuter(node_features, mask=graph.mask, hard=not training, tau=tau)
        graph_pred = self.decode(graph_emb, perm, graph.mask)
        return graph_pred, perm, mu, logvar


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
        message_input_dim = 2 * (hparams["num_node_features"] + 1) + hparams["num_edge_features"] + 1
        self.fc_in = Linear(message_input_dim, hparams["graph_encoder_hidden_dim"])
        self.layer_norm = LayerNorm(hparams["graph_encoder_hidden_dim"])
        self.dropout = Dropout(0.1)

    def add_emb_node_and_feature(self, node_features, edge_features, mask):
        node_dim, edge_dim = node_features.size(-1), edge_features.size(-1)
        node_features = pad(node_features, (0, 1, 1, 0))
        edge_features = pad(edge_features, (0, 1, 1, 0, 1, 0))
        edge_features[:, 0, :, edge_dim] = 1
        edge_features[:, :, 0, edge_dim] = 1
        mask = pad(mask, (1, 0), value=1)
        return node_features, edge_features, mask

    def init_message_matrix(self, node_features, edge_features, mask):
        node_features, edge_features, mask = self.add_emb_node_and_feature(node_features, edge_features, mask)
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        batch_size, num_nodes = node_features.size(0), node_features.size(1)
        node_features_combined = torch.cat(
            (node_features.unsqueeze(2).repeat(1, 1, num_nodes, 1),
             node_features.unsqueeze(1).repeat_interleave(num_nodes, dim=1)),
            dim=-1)
        x = torch.cat((edge_features, node_features_combined), dim=-1)
        x = self.layer_norm(self.dropout(self.fc_in(x)))
        return x, edge_mask

    def read_out_message_matrix(self, x):
        node_features = torch.diagonal(x, dim1=1, dim2=2).transpose(1, 2)
        graph_emb, node_features = node_features[:, 0], node_features[:, 1:]
        return graph_emb, node_features

    def forward(self, node_features, edge_features, mask):
        x, edge_mask = self.init_message_matrix(node_features, edge_features, mask)
        x = self.graph_transformer(x, mask=edge_mask)
        graph_emb, node_features = self.read_out_message_matrix(x)

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
        message_input_dim = hparams["graph_decoder_hidden_dim"] + 2 * hparams["graph_decoder_pos_emb_dim"]
        self.fc_in = Linear(message_input_dim, hparams["graph_decoder_hidden_dim"])
        self.node_fc_out = Linear(hparams["graph_decoder_hidden_dim"], hparams["num_node_features"])
        self.edge_fc_out = Linear(hparams["graph_decoder_hidden_dim"], hparams["num_edge_features"])
        self.dropout = Dropout(0.1)
        self.layer_norm = LayerNorm(hparams["graph_decoder_hidden_dim"])

    def init_message_matrix(self, graph_emb, perm, num_nodes):
        batch_size= graph_emb.size(0)

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
        return x

    def read_out_message_matrix(self, x):
        num_nodes = x.size(1)
        node_features = torch.diagonal(x, dim1=1, dim2=2).transpose(1, 2)
        node_features = self.node_fc_out(node_features)
        edge_features = self.edge_fc_out(x)
        self_edge_mask = torch.eye(num_nodes, num_nodes, device=node_features.device).bool().unsqueeze(-1)
        edge_features.masked_fill_(self_edge_mask, 0)
        edge_features = (edge_features + edge_features.permute(0, 2, 1, 3)) / 2
        return node_features, edge_features

    def forward(self, graph_emb, perm, mask):
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        x = self.init_message_matrix(graph_emb, perm, num_nodes=mask.size(1))
        x = self.graph_transformer(x, mask=edge_mask)
        node_features, edge_features = self.read_out_message_matrix(x)
        return node_features, edge_features


class Permuter(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.scoring_fc = Linear(hparams["graph_decoder_hidden_dim"], 1)

    def score(self, x, mask):
        scores = self.scoring_fc(x)
        fill_value = scores.min().item() - 1
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, fill_value)
        return scores

    def soft_sort(self, scores, hard, tau):
        scores_sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - scores_sorted).abs().neg() / tau
        perm = pairwise_diff.softmax(-1)
        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
        return perm

    def mask_perm(self, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    def forward(self, node_features, mask, hard=False, tau=1.0):
        # add noise to break symmetry
        node_features = node_features + torch.randn_like(node_features) * 0.05
        scores = self.score(node_features, mask)
        perm = self.soft_sort(scores, hard, tau)
        perm = perm.transpose(2, 1)
        perm = self.mask_perm(perm, mask)
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
    def __init__(self, hparams):
        super().__init__()
        self.d_in = hparams["graph_encoder_hidden_dim"]
        self.d_out = hparams["emb_dim"]
        self.vae = hparams["vae"]
        if self.vae:
            self.w = Linear(self.d_in, 2 * self.d_out)
        else:
            self.w = Linear(self.d_in, self.d_out)

    def forward(self, x):
        x = self.w(relu(x))
        if self.vae:
            mu = x[:, :self.d_out]
            logvar = x[:, self.d_out:]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = mu + eps * std
            return x, mu, logvar
        else:
            return x, None, None


class BottleNeckDecoder(torch.nn.Module):
    def __init__(self, hparams):
        self.d_in = hparams["emb_dim"]
        self.d_out = hparams["graph_decoder_hidden_dim"]
        super().__init__()
        self.w = Linear(self.d_in, self.d_out)

    def forward(self, x):
        x = self.w(x)
        return x


class PropertyPredictor(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()
        d_in = hparams["emb_dim"]
        d_hid = hparams["property_predictor_hidden_dim"]
        d_out = hparams["num_properties"]
        self.w_1 = Linear(d_in, d_hid)
        self.w_2 = Linear(d_hid, d_hid)
        self.w_3 = Linear(d_hid, d_out)
        self.layer_norm1 = LayerNorm(d_hid)
        self.layer_norm2 = LayerNorm(d_hid)
        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm1(self.dropout(relu(self.w_1(x))))
        x = self.layer_norm2(self.dropout(relu(self.w_2(x))))
        x = self.w_3(x)
        return x