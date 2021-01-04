import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem.rdmolops import GetDistanceMatrix
from torch_geometric.data import Data
from torch_geometric.transforms import ToDense
from torch_geometric.utils import to_dense_batch, to_dense_adj

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'H']
CHARGE_LIST = [-1, -2, 1, 2, 0]
HYBRIDIZATION_TYPE_LIST = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                           Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                           Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                           Chem.rdchem.HybridizationType.UNSPECIFIED]
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
HS_LIST = [0, 1, 2, 3]
NUM_ELEMENTS = len(ELEM_LIST)
NUM_CHARGES = len(CHARGE_LIST)
NUM_HS = len(HS_LIST)
NUM_ATOMS_MEAN = 23.101
NUM_ATOMS_STD = 6.652

MEAN_DISTANCE = 4.814012207760507
STD_DISTANCE = 2.991864705281403


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
        other_attr = {key: [] for key in graph_list[0].keys if key not in ['x', 'edge_index', 'edge_attr', 'distance_matrix']}
        for graph in graph_list:
            x_, mask_ = to_dense_batch(graph.x, max_num_nodes=max_num_nodes)
            adj_ = to_dense_adj(graph.edge_index, edge_attr=graph.edge_attr, max_num_nodes=max_num_nodes)
            adj_ = add_empty_edge_type(adj_)
            dm = torch.ones(max_num_nodes, max_num_nodes) * -100
            dm[:graph.num_nodes, :graph.num_nodes] = graph.distance_matrix
            dm = dm.unsqueeze(0).unsqueeze(-1)
            adj_ = torch.cat((adj_, dm), axis=-1)
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


class MolecularGraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, max_num_nodes, num_eval_samples, num_samples_per_epoch,
                 num_samples_per_epoch_inc, num_workers=1, debug=False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_num_nodes = max_num_nodes
        self.num_eval_samples = num_eval_samples
        self.num_samples_per_epoch = num_samples_per_epoch
        self.num_samples_per_epoch_inc = num_samples_per_epoch_inc
        self.num_workers = num_workers
        self.debug = debug
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None

    def setup(self, stage):
        num_smiles = 10000000 if self.debug else None
        smiles_df = pd.read_csv(self.data_path, nrows=num_smiles, compression="gzip")
        self.train_smiles_df = smiles_df.iloc[self.num_eval_samples:]
        self.eval_smiles_df = smiles_df.iloc[:self.num_eval_samples]

    def train_dataloader(self):
        print(self.max_num_nodes)
        smiles_df = self.train_smiles_df[self.train_smiles_df.num_atoms <= self.max_num_nodes]
        smiles_df = smiles_df.sample(frac=1.0).reset_index(drop=True)
        smiles_df = smiles_df.iloc[:self.num_samples_per_epoch]
        train_dataset = MolecularGraphDatasetFromSmilesDataFrame(df=smiles_df)
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            shuffle=True
        )
        #self.max_num_nodes += 2
        self.num_samples_per_epoch += self.num_samples_per_epoch_inc
        return DenseGraphDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        smiles_df = self.eval_smiles_df[self.eval_smiles_df.num_atoms <= self.max_num_nodes].reset_index(drop=True)
        eval_dataset = MolecularGraphDatasetFromSmilesDataFrame(df=smiles_df)
        eval_sampler = DistributedSampler(
            dataset=eval_dataset,
            shuffle=False
        )
        return DenseGraphDataLoader(
            dataset=eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=eval_sampler,
        )


class MolecularGraphDatasetFromSmiles(Dataset):
    def __init__(self, smiles_list, randomize_smiles=True):
        super().__init__()
        self.smiles = smiles_list
        self.randomize_smiles = randomize_smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        if self.randomize_smiles:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True)
        mol = Chem.MolFromSmiles(smiles)
        graph = MolecularGraph.from_mol(mol)
        graph.distance_matrix = torch.from_numpy((GetDistanceMatrix(mol) - MEAN_DISTANCE) / STD_DISTANCE).float()
        return graph


class MolecularGraphDatasetFromSmilesDataFrame(MolecularGraphDatasetFromSmiles):
    def __init__(self, df, randomize_smiles=True):
        super().__init__(smiles_list=df.smiles.tolist())
        self.df = df
        self.randomize_smiles = randomize_smiles
        self.properties = [
            "num_atoms", "logp", "mr", "balabanj", "num_h_acceptors", "num_h_donors", "num_valence_electrons", "tpsa"
        ]

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx].smiles
        props = torch.from_numpy(self.df.loc[idx, self.properties].to_numpy().astype(np.float32))
        props[0] = (props[0] - NUM_ATOMS_MEAN) / NUM_ATOMS_STD
        props = props.unsqueeze(0)
        if self.randomize_smiles:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True)
        mol = Chem.MolFromSmiles(smiles)
        graph = MolecularGraph.from_mol(mol)
        graph.distance_matrix = torch.from_numpy((GetDistanceMatrix(mol) - MEAN_DISTANCE) / STD_DISTANCE).float()
        graph.molecular_properties = props
        return graph


class MolecularGraph(Data):
    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        graph = cls.from_mol(mol=mol)
        #graph.x = add_embedding_node(graph.x)
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
    atom_feat.extend(one_hot_encoding(atom.GetNumExplicitHs(), HS_LIST))
    atom_feat.extend(one_hot_encoding(atom.GetNumImplicitHs(), HS_LIST))
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
    num_hs = torch.where(node_features[:, NUM_ELEMENTS + NUM_CHARGES: NUM_ELEMENTS + NUM_CHARGES + NUM_HS] == 1)[1].tolist()
    num_hs = [HS_LIST[idx] for idx in num_hs]
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
        atom.SetNumExplicitHs(num_hs[i])
        mol.AddAtom(atom)
    for bond, bond_type in bonds.items():
        mol.AddBond(bond[0], bond[1], bond_type)
    m = mol.GetMol()
    try:
        Chem.SanitizeMol(m)
        return m
    except:
        return None


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


def add_embedding_node(x):
    shape = x.shape
    x = torch.cat((x, torch.zeros(shape[0], 1)), dim=1)
    x = torch.cat((torch.Tensor(shape[1] * [0] + [1]).unsqueeze(0), x), dim=0)
    return x

def add_empty_edge_type(adj):
    shape = adj.shape
    if adj.dim() == 3:
        adj = adj.unsqueeze(-1)
    mask = torch.all(adj == 0, axis=-1)
    empty_edge = torch.zeros((shape[0], shape[1], shape[2], 1)).type_as(adj)
    empty_edge[mask] = 1
    adj = torch.cat((empty_edge, adj), axis=-1)
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

