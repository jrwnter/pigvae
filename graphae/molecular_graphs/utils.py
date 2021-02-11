from random import shuffle
import torch
from rdkit import Chem


def suffle_and_filter_qm9_sdf(input_sdf_path="gdb9.sdf", output_sdf_path="qm9.sdf"):
    suppl = Chem.SDMolSupplier(input_sdf_path, removeHs=False)
    sdf_writer = Chem.SDWriter(output_sdf_path)
    mols = []
    for i in range(len(suppl)):
        mol = suppl[int(i)]
        if mol is not None:
            mols.append(mol)
    shuffle(mols)
    for mol in mols:
        sdf_writer.write(mol)

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