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
            non_linearity=hparams["nonlin"]
        )

    def forward(self, node_features, adj, mask=None):
        mol_emb = self.graph_encoder(node_features, adj, mask)
        return mol_emb


class Decoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fnn = FNN(
            input_dim=hparams["emb_dim"],
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

    def forward(self, z):
        x = self.fnn(z)
        edge_logits = self.edge_predictor(x)
        node_logits = self.node_predictor(x)
        return node_logits, edge_logits


class Descriminator(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.linear = torch.nn.Linear(hparams["emb_dim"], hparams["emb_dim"] )
        self.fnn = FNN(
            #input_dim=hparams["graph_encoder_num_layers"] * hparams["node_dim"],
            input_dim=hparams["emb_dim"] + 1,
            hidden_dim=512,
            output_dim=1,
            num_layers=3,
            non_linearity=hparams["nonlin"],
            batch_norm=False,
            dropout=0.2
        )

    def forward(self, node, adj, mask=None):
        h = self.encoder(node, adj, mask)
        c = self.linear(h)
        sim = torch.abs(c.unsqueeze(0) - c.unsqueeze(1)).sum(dim=-1)
        sim = torch.exp(-sim).sum(-1).unsqueeze(-1)
        x = torch.cat((h, sim), dim=1)
        x = self.fnn(x).squeeze()
        return x, h


class GraphVAEGAN(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.generator = Decoder(hparams)
        self.discriminator = Descriminator(hparams)
        hparams2 = hparams.copy()
        hparams2["emb_dim"] *= 2
        self.encoder = Encoder(hparams2)

    def forward(self, node_features, adj, mask=None):
        mol_emb = self.encoder(node_features, adj, mask)
        mu, log_var = mol_emb[:, :128], mol_emb[:, 128:]
        mol_emb = reparameterize(mu, log_var)
        node_features_, adj_ = self.generator(mol_emb)

        return node_features_, adj_, mol_emb


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


