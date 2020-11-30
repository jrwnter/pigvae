import torch
from torch.nn import Linear
from graphae.fully_connected import FNN


class PropertyPredictor(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        """self.fnn = FNN(
            input_dim=hparams["emb_dim"],
            hidden_dim=hparams["property_predictor_hidden_dim"],
            output_dim=hparams["num_properties"],
            num_layers=hparams["property_predictor_num_layers"],
            non_linearity=hparams["nonlin"],
            batch_norm=hparams["batch_norm"],
        )"""
        self.fnn = Linear(
            in_features=hparams["emb_dim"],
            out_features=hparams["num_properties"]
        )

    def forward(self, x):
        y = self.fnn(x)
        return y
