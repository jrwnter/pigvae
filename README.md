# Permutation-Invariant Variational Autoencoder (PIGVAE)

Implementation of the Paper "Permutation-Invariant Variational Autoencoder for Graph-Level Representation Learning" by Robin Winter, Frank Noe and Djork-Arne Clevert.

## Installing

Prerequisites:
- python 3.8
- pytorch=1.7
- pytorch geometric
- pytorch-lightning=1.3.1
- rdkit
- numpy
- networkx

Train on synthetic random graphs, specifying a graph family and parameters:
- binomial graphs, on 4 gpus, with parameter p in (0.4, 0.6):
    python pigvae/syntetic_graphs/main.py --progress_bar -i 1 -g 4 --graph_family binomial -b 64 --p_min 0.4 --p_max 0.6
- mix of 10 different graphs families, on 4 gpus, with parameter p in (0.4, 0.6):
    python pigvae/syntetic_graphs/main.py --progress_bar -i 1 -g 4 --graph_family all -b 64


