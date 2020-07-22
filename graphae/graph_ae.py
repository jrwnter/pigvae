import torch
from graphae import encoder, decoder
from graphae.fully_connected import FNN


class Encoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.graph_encoder = encoder.GraphEncoder(
            input_dim=hparams["num_atom_features"],
            hidden_dim=hparams["graph_encoder_hidden_dim"],
            node_dim=hparams["node_dim"],
            emb_dim=hparams["emb_dim"],
            num_layers=hparams["graph_encoder_num_layers"],
            batch_norm=hparams["batch_norm"],
            non_linearity="lrelu"
        )

    def forward(self, node_features, adj, mask):
        mol_emb = self.graph_encoder(node_features, adj, mask)
        return mol_emb


class Decoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.meta_node_decoder = decoder.MetaNodeDecoder(
            num_nodes=hparams["max_num_nodes"],
            emb_dim=hparams["emb_dim"],
            meta_node_dim=hparams["meta_node_dim"],
            hidden_dim=hparams["meta_node_decoder_hidden_dim"],
            num_layers=hparams["meta_node_decoder_num_layers"],
            batch_norm=hparams["batch_norm"],
        )
        self.edge_predictor = decoder.EdgePredictor(
            num_nodes=hparams["max_num_nodes"],
            meta_node_dim=hparams["meta_node_dim"],
            hidden_dim=hparams["edge_predictor_hidden_dim"],
            num_layers=hparams["edge_predictor_num_layers"],
            batch_norm=hparams["batch_norm"],
        )
        self.node_predictor = decoder.NodePredictor(
            num_nodes=hparams["max_num_nodes"],
            meta_node_dim=hparams["meta_node_dim"],
            hidden_dim=hparams["node_decoder_hidden_dim"],
            num_layers=hparams["node_decoder_num_layers"],
            batch_norm=hparams["batch_norm"],
            num_node_features=hparams["num_atom_features"]
        )

    def forward(self, emb):
        meta_node_emb = self.meta_node_decoder(emb)
        adj = self.edge_predictor(meta_node_emb)
        node_features = self.node_predictor(meta_node_emb)
        mask, node_features = node_features[:, :, -1], node_features[:, :, :-1]
        return node_features, adj, mask


class SideTaskPredictor(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fnn = FNN(
            input_dim=hparams["emb_dim"],
            hidden_dim=1024,
            output_dim=2,
            num_layers=4,
            non_linearity="elu",
            batch_norm=True,
        )

    def forward(self, emb):
        return self(emb)


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.predictor = SideTaskPredictor(hparams)

    def forward(self, node_features, adj, mask, grad_mode="encode"):
        mol_emb = self.encoder(node_features, adj, mask)
        node_features_, adj_, mask_ = self.decoder(mol_emb.detach())
        mol_emb_ = self.encoder(node_features_, adj_, mask_)
        output = {
            "node_features_real": node_features,
            "node_features_pred": node_features_,
            "adj_real": adj,
            "adj_pred": adj_,
            "mask_real": mask,
            "mask_pred": mask_,
            "mol_emb_real": mol_emb,
            "mol_emb_pred": mol_emb_,
        }

        return output

