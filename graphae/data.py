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


class MolecularGraphDataset(Dataset):
    def __init__(self, smiles_list, num_nodes):
        super().__init__()
        self.smiles = smiles_list
        self.dense_transform = ToDense(num_nodes=num_nodes)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        graph = MolecularGraph.from_smiles(smiles)
        graph = self.dense_transform(graph)
        return graph.x, graph.adj, graph.mask


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
        edge_index = get_edge_index(mol)
        x = get_node_features(mol)
        graph_props = {
            "edge_index": edge_index,
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


def get_node_features(mol):
    return torch.Tensor([one_hot_atom_features(a) for a in mol.GetAtoms()])


def one_hot_atom_features(atom):
    atom_feat = []
    atom_feat.extend(one_hot_encoding(atom.GetSymbol(), ELEM_LIST))
    atom_feat.extend(one_hot_encoding(atom.GetFormalCharge(), CHARGE_LIST))
    atom_feat.extend(one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE_LIST))
    atom_feat.extend([atom.GetIsAromatic()])
    return atom_feat

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