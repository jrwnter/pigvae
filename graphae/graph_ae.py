import torch
from graphae import encoder, decoder


class GraphAE(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Encoder
        self.graph_encoder = encoder.GraphEncoder(
            input_dim=hparams["num_atom_features"],
            output_dim=hparams["node_dim"],
            hidden_dim=hparams["graph_encoder_hidden_dim"],
            num_layers=hparams["graph_encoder_num_layers"],
            batch_norm=False,
            non_linearity="elu"
        )
        self.node_aggregator = encoder.NodeAggregator(
            node_dim=hparams["node_dim"],
            emb_dim=hparams["emb_dim"],
            hidden_dim=hparams["node_aggregator_hidden_dim"],
            num_layers=hparams["node_aggregator_num_layers"],
            num_nodes=hparams["max_num_nodes"],
            batch_norm=False
        )
        # Decoder
        self.meta_node_decoder = decoder.MetaNodeDecoder(
            num_nodes=hparams["max_num_nodes"],
            emb_dim=hparams["emb_dim"],
            meta_node_dim=hparams["meta_node_dim"],
            hidden_dim=hparams["meta_node_decoder_hidden_dim"],
            num_layers=hparams["meta_node_decoder_num_layers"],
            batch_norm=False
        )
        self.edge_predictor = decoder.EdgePredictor(
            num_nodes=hparams["max_num_nodes"],
            meta_node_dim=hparams["meta_node_dim"],
            hidden_dim=hparams["edge_predictor_hidden_dim"],
            num_layers=hparams["edge_predictor_num_layers"],
            batch_norm=False
        )
        self.node_predictor = decoder.NodePredictor(
            num_nodes=hparams["max_num_nodes"],
            meta_node_dim=hparams["meta_node_dim"],
            hidden_dim=hparams["node_decoder_hidden_dim"],
            num_layers=hparams["node_decoder_num_layers"],
            batch_norm=False,
            num_node_features=hparams["num_atom_features"]
        )

    def forward(self, node_features, adj, mask):
        node_emb = self.graph_encoder(node_features, adj, mask)
        mol_emb = self.node_aggregator(node_emb, mask)
        meta_node_emb = self.meta_node_decoder(mol_emb)
        adj = self.edge_predictor(meta_node_emb)
        node_features = self.node_predictor(meta_node_emb)
        mask, node_features = node_features[:, :, 0], node_features[:, :, 1:]
        mask = torch.sigmoid(mask)

        node_emb_gen = self.graph_encoder(node_features, adj, mask)
        mol_emb_gen = self.node_aggregator(node_emb_gen, mask)

        return mol_emb, mol_emb_gen, adj, mask, node_features


