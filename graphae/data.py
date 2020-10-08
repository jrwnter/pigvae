import torch
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import ToDense

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'H']
CHARGE_LIST = [-1, -2, 1, 2, 0]
HYBRIDIZATION_TYPE_LIST = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                           Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                           Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                           Chem.rdchem.HybridizationType.UNSPECIFIED]
HS_LIST = [0, 1, 2, 3]
NUM_ELEMENTS = len(ELEM_LIST)


class MolecularGraphDatasetFromSmiles(Dataset):
    def __init__(self, smiles_list, num_nodes):
        super().__init__()
        self.smiles = smiles_list
        self.dense_transform = ToDense(num_nodes=num_nodes)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        sparse_graph = MolecularGraph.from_smiles(smiles)
        dense_graph = self.dense_transform(sparse_graph.clone())
        dense_graph.x = dense_graph.x.unsqueeze(0)
        dense_graph.adj = dense_graph.adj.unsqueeze(0)
        dense_graph.mask = dense_graph.mask.unsqueeze(0)
        return sparse_graph, dense_graph


class MolecularGraphDataset(Dataset):
    def __init__(self, graphs, noise):
        super().__init__()
        self.graphs = graphs
        self.noise = noise

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        x = torch.from_numpy(graph[:, :11])
        adj = torch.from_numpy(graph[:, 24:40])
        mask = torch.from_numpy(graph[:, 40])
        if self.noise:
            x, adj, mask = add_noise(x.detach().clone(), adj.detach().clone(), mask.detach().clone())
        return x, adj, mask


class MolecularGraph(Data):
    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        graph = cls.from_mol(mol=mol)
        return graph

    @classmethod
    def from_mol(cls, mol):
        graph_props, follow_batch = cls.get_graph_properties(mol)
        graph = Data(**graph_props)
        return graph

    @staticmethod
    def get_graph_properties(mol):
        edge_attr, edge_index = get_edge_attr_index(mol)
        x = get_node_features(mol)
        graph_props = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "x": x,
        }
        follow_batch = ["edge_index"]
        return graph_props, follow_batch


def one_hot_encoding(x, lst):
    if x not in lst:
        x = lst[-1]
    return list(map(lambda s: x == s, lst))


def get_edge_index(mol):
    edge_index = []
    for b in mol.GetBonds():
        edge_index.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
    edge_index = torch.LongTensor(edge_index).permute(1, 0)
    # both direction
    edge_index = torch.stack(
        (torch.cat((edge_index[0], edge_index[1])),
         torch.cat((edge_index[1], edge_index[0])))
    )
    return edge_index


def get_edge_attr_index(mol):
    edge_attr = []
    edge_index = []
    for b in mol.GetBonds():
        edge_attr.append(one_hot_bond_features(b))
        edge_index.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
    edge_attr = torch.Tensor(edge_attr)
    edge_index = torch.LongTensor(edge_index).permute(1, 0)
    # both direction
    edge_attr = torch.cat((edge_attr, edge_attr))
    edge_index = torch.stack(
        (torch.cat((edge_index[0], edge_index[1])),
         torch.cat((edge_index[1], edge_index[0])))
    )
    return edge_attr, edge_index


def get_node_features(mol):
    return torch.Tensor([one_hot_atom_features(a) for a in mol.GetAtoms()])


def one_hot_atom_features(atom):
    atom_feat = []
    atom_feat.extend(one_hot_encoding(atom.GetSymbol(), ELEM_LIST))
    atom_feat.extend(one_hot_encoding(atom.GetFormalCharge(), CHARGE_LIST))
    atom_feat.extend(one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE_LIST))
    #atom_feat.extend([atom.GetIsAromatic()])
    return atom_feat


def one_hot_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feat = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing()]
    return bond_feat


def rdkit_mol_from_graph(graph):
    graph = graph.to("cpu")
    num_atoms = len(graph.x)
    atom_type = torch.where(graph.x[:, :NUM_ELEMENTS] == 1)[1].tolist()
    atom_type = [ELEM_LIST[idx] for idx in atom_type]
    charge = torch.where(graph.x[:, NUM_ELEMENTS:NUM_ELEMENTS + 5] == 1)[1].tolist()
    charge = [CHARGE_LIST[idx] for idx in charge]
    hybridization = torch.where(graph.x[:, NUM_ELEMENTS + 5: NUM_ELEMENTS + 5 + 7] == 1)[1].tolist()
    hybridization = [HYBRIDIZATION_TYPE_LIST[idx] for idx in hybridization]
    is_aromatic = graph.x[:, -1].bool().tolist()
    bonds = []
    for i in range(graph.edge_index.shape[1]):
        edge_idx = tuple(graph.edge_index[:, i].tolist())
        edge_idx_reversed = (edge_idx[1], edge_idx[0])
        if (edge_idx not in bonds) & (edge_idx_reversed not in bonds):
            bonds.append(edge_idx)
    mol = Chem.RWMol()
    Chem.SanitizeMol(mol)
    for i in range(num_atoms):
        atom = Chem.Atom(atom_type[i])
        atom.SetIsAromatic(is_aromatic[i])
        atom.SetFormalCharge(charge[i])
        atom.SetHybridization(hybridization[i])
        mol.AddAtom(atom)
    for bond in bonds:
        mol.AddBond(bond[0], bond[1])
    mol = mol.GetMol()
    return mol


def add_noise(x, adj, mask, std=0.01):
    noise_x = torch.randn_like(x).type_as(x) * std
    noise_x[(x + noise_x > 1.0) | (x + noise_x < 0.0)] *= -1
    x = x + noise_x

    noise_adj = torch.randn_like(adj).type_as(adj) * std
    noise_adj[(adj + noise_adj > 1.0) | (adj + noise_adj < 0.0)] *= -1
    adj = adj + noise_adj

    noise_mask = torch.randn_like(mask).type_as(mask) * std
    noise_mask[(mask + noise_mask > 1.0) | (mask + noise_mask < 0.0)] *= -1
    mask = mask + noise_mask

    return x, adj, mask


def add_empty_node_type(nodes):
    shape = nodes.shape
    mask = torch.all(nodes == 0, axis=-1)
    empty_node = torch.zeros((shape[0], shape[1], 1)).type_as(nodes)
    empty_node[mask] = 1
    nodes = torch.cat((nodes, empty_node), axis=-1)
    return nodes


def add_empty_edge_type(adj):
    shape = adj.shape
    if adj.dim() == 3:
        adj = adj.unsqueeze(-1)
    mask = torch.all(adj == 0, axis=-1)
    empty_edge = torch.zeros((shape[0], shape[1], shape[2], 1)).type_as(adj)
    empty_edge[mask] = 1
    adj = torch.cat((adj, empty_edge), axis=-1)
    return adj


def batch_to_dense(batch, num_nodes):
    dense_transform = ToDense(num_nodes=num_nodes)
    dense_graphs = [dense_transform(batch) for batch in batch.to_data_list()]
    x = []
    adj = []
    for graph in dense_graphs:
        x.append(graph.x)
        adj.append(graph.adj)
    x = torch.stack(x, dim=0)
    adj = torch.stack(adj, dim=0)
    return x, adj

