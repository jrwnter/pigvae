import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem.rdmolops import GetDistanceMatrix, Get3DDistanceMatrix
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj


ELEM_LIST = ['C', 'N', 'O', 'F', 'H']
CHARGE_LIST = [-1, 1, 0]
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
NUM_ELEMENTS = len(ELEM_LIST)
NUM_CHARGES = len(CHARGE_LIST)

NUM_ATOMS_MEAN = 23.101
NUM_ATOMS_STD = 6.652
MEAN_TOP_DISTANCE = 4.814012207760507
STD_TOP_DISTANCE = 2.991864705281403
MEAN_EUCL_DISTANCE = 3.0955
STD_EUCL_DISTANCE = 1.5164


class MolecularGraphDataModule(pl.LightningDataModule):
    def __init__(self, sdf_path, batch_size, num_eval_samples, num_workers=1, distributed_sampler=True):
        super().__init__()
        self.sdf_path = sdf_path
        self.batch_size = batch_size
        self.num_eval_samples = num_eval_samples
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None

    def train_dataloader(self):
        self.train_dataset = MolecularGraphDatasetFromSDF(self.sdf_path, offset=self.num_eval_samples)
        if self.distributed_sampler:
            train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                shuffle=True
            )
        else:
            train_sampler = None
        return DenseGraphDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        self.eval_dataset = MolecularGraphDatasetFromSDF(self.sdf_path, num_mols=self.num_eval_samples)
        if self.distributed_sampler:
            eval_sampler = DistributedSampler(
                dataset=self.eval_dataset,
                shuffle=False
            )
        else:
            eval_sampler = None
        return DenseGraphDataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=eval_sampler,
        )


class DenseGraphBatch(Data):
    def __init__(self, node_features, edge_features, mask, **kwargs):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        for key, item in kwargs.items():
            setattr(self, key, item)

    @classmethod
    def from_sparse_graph_list(cls, graph_list):
        max_num_nodes = max([len(graph.x) for graph in graph_list])
        x = []
        mask = []
        adj = []
        other_attr = {key: [] for key in graph_list[0].keys
                      if key not in ['x', 'edge_index', 'edge_attr', 'top_distance_matrix', 'eucl_distance_matrix']}
        for graph in graph_list:
            x_, mask_ = to_dense_batch(graph.x, max_num_nodes=max_num_nodes)
            adj_ = to_dense_adj(graph.edge_index, edge_attr=graph.edge_attr, max_num_nodes=max_num_nodes)
            adj_ = add_empty_edge_type(adj_)
            top_dm = torch.ones(max_num_nodes, max_num_nodes) * -100
            eucl_dm = torch.ones(max_num_nodes, max_num_nodes) * -100
            top_dm[:graph.num_nodes, :graph.num_nodes] = graph.top_distance_matrix
            eucl_dm[:graph.num_nodes, :graph.num_nodes] = graph.eucl_distance_matrix
            top_dm = top_dm.unsqueeze(0).unsqueeze(-1)
            eucl_dm = eucl_dm.unsqueeze(0).unsqueeze(-1)
            adj_ = torch.cat((adj_, top_dm, eucl_dm), axis=-1)
            x.append(x_)
            mask.append(mask_)
            adj.append(adj_)
            for key in other_attr.keys():
                attr = graph[key]
                # add batch dim
                if attr.size(0) != 0:
                    attr.unsqueeze(0)
                other_attr[key].append(attr)
        x = torch.cat(x, dim=0)
        mask = torch.cat(mask, dim=0)
        # set self edges to 0
        adj = torch.cat(adj, dim=0)
        self_edge_mask = torch.eye(max_num_nodes, max_num_nodes).bool().unsqueeze(-1)
        adj.masked_fill_(self_edge_mask, 0)
        for key in other_attr.keys():
            other_attr[key] = torch.cat(other_attr[key])
        return cls(node_features=x, edge_features=adj, mask=mask, **other_attr)

    def __repr__(self):
        repr_list = ["{}={}".format(key, list(value.shape)) for key, value in self.__dict__.items()]
        return "DenseGraphBatch({})".format(", ".join(repr_list))


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=lambda data_list: DenseGraphBatch.from_sparse_graph_list(data_list), **kwargs)


class MolecularGraphDatasetFromSDF(Dataset):
    def __init__(self, sdf_path, randomize_smiles=True, offset=None, num_mols=None):
        super().__init__()
        self.suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        self.offset=offset
        self.num_mols=num_mols

    def __len__(self):
        if self.num_mols is not None:
            length = self.num_mols
        elif self.offset is not None:
            length = len(self.suppl) - self.offset
        else:
            length = len(self.suppl)
        return length

    def __getitem__(self, idx):
        if self.offset is not None:
            mol = self.suppl[int(idx + self.offset)]
        else:
            mol = self.suppl[int(idx)]
        graph = MolecularGraph.from_mol(mol)
        return graph


class MolecularGraph(Data):
    @classmethod
    def from_mol(cls, mol):
        graph_props, follow_batch = cls.get_graph_properties(mol)
        mol_props = cls.get_mol_properties(mol)
        top_distance_matrix = cls.get_topological_distance_matrix(mol)
        eucl_distance_matrix = cls.get_euclidean_distance_matrix(mol)
        graph = Data(**graph_props,
                     properties=mol_props,
                     top_distance_matrix=top_distance_matrix,
                     eucl_distance_matrix=eucl_distance_matrix
                     )
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

    @staticmethod
    def get_mol_properties(mol):
        num_atoms = mol.GetNumAtoms()
        num_atoms_normalized = (num_atoms - NUM_ATOMS_MEAN) / NUM_ATOMS_STD
        properties = torch.Tensor([num_atoms_normalized])
        return properties

    @staticmethod
    def get_topological_distance_matrix(mol):
        dm = torch.from_numpy(GetDistanceMatrix(mol)).float()
        dm = (dm - MEAN_TOP_DISTANCE) / STD_TOP_DISTANCE
        return dm

    @staticmethod
    def get_euclidean_distance_matrix(mol):
        dm = torch.from_numpy(Get3DDistanceMatrix(mol)).float()
        dm = (dm - MEAN_EUCL_DISTANCE) / STD_EUCL_DISTANCE
        return dm


def one_hot_encoding(x, lst):
    if x not in lst:
        x = lst[-1]
    return list(map(lambda s: x == s, lst))


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
    atom_feat.extend([int(atom.GetIsAromatic())])
    atom_feat.extend([int(atom.IsInRing())])
    return atom_feat


def one_hot_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feat = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing(),
        bond.IsInRingSize(3),
        bond.IsInRingSize(4),
        bond.IsInRingSize(5),
        bond.IsInRingSize(6),
    ]
    return bond_feat


def rdkit_mol_from_graph(node_features, edge_features):
    num_atoms = len(node_features)
    atom_type = torch.where(node_features[:, :NUM_ELEMENTS] == 1)[1].tolist()
    atom_type = [ELEM_LIST[idx] for idx in atom_type]
    charge = torch.where(node_features[:, NUM_ELEMENTS:NUM_ELEMENTS + NUM_CHARGES] == 1)[1].tolist()
    charge = [CHARGE_LIST[idx] for idx in charge]
    bonds = {}
    edge_index = []
    edge_attr = []
    for j in range(1, 5):
        idx1, idx2 = torch.where(edge_features[:, :, j] == 1)
        edge_index.append(torch.stack((idx1, idx2), dim=1))
        edge_attr.append(torch.Tensor(len(idx1) * [j - 1]))
    edge_index = torch.cat(edge_index, dim=0).numpy()
    edge_attr = torch.cat(edge_attr).numpy()
    for i in range(len(edge_index)):
        edge = tuple(edge_index[i])
        edge_reversed = (edge[1], edge[0])
        if edge_reversed not in bonds:
            bonds[(int(edge[0]), int(edge[1]))] = BOND_LIST[int(edge_attr[i])]
    mol = Chem.RWMol()
    for i in range(num_atoms):
        atom = Chem.Atom(atom_type[i])
        atom.SetFormalCharge(charge[i])
        mol.AddAtom(atom)
    for bond, bond_type in bonds.items():
        mol.AddBond(bond[0], bond[1], bond_type)
    m = mol.GetMol()
    try:
        Chem.SanitizeMol(m)
        return m
    except:
        return None


def add_empty_edge_type(adj):
    shape = adj.shape
    if adj.dim() == 3:
        adj = adj.unsqueeze(-1)
    mask = torch.all(adj == 0, axis=-1)
    empty_edge = torch.zeros((shape[0], shape[1], shape[2], 1)).type_as(adj)
    empty_edge[mask] = 1
    adj = torch.cat((empty_edge, adj), axis=-1)
    return adj
