import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.transforms import ToDense

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
NUM_ATOMS_MEAN = 23.101
NUM_ATOMS_STD = 6.652


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
        num_smiles = 100000 if self.debug else None
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
        self.max_num_nodes += 2
        self.num_samples_per_epoch += self.num_samples_per_epoch_inc
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        eval_dataset = MolecularGraphDatasetFromSmilesDataFrame(df=self.eval_smiles_df)
        eval_sampler = DistributedSampler(
            dataset=eval_dataset,
            shuffle=False
        )
        return DataLoader(
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

    def dense_edge_index(self, graph):
        num_elements = graph.x.size(0)
        edge_index = torch.combinations(torch.arange(num_elements), 2).transpose(1, 0)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=0).transpose(1, 0).reshape(-1, 2).transpose(1, 0)
        return edge_index

    def dense_edge_attr(self, graph):
        idx1, idx2 = torch.where(
            (graph.dense_edge_index.unsqueeze(2) == graph.edge_index.unsqueeze(1)).all(dim=0))
        dense_edge_attr = torch.cat((
            torch.zeros(graph.dense_edge_index.size(1), graph.edge_attr.size(1)),
            torch.ones(graph.dense_edge_index.size(1), 1)), dim=-1)
        dense_edge_attr[idx1] = torch.cat((graph.edge_attr, torch.zeros(graph.edge_attr.size(0), 1)), dim=-1)[idx2]
        return dense_edge_attr

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        if self.randomize_smiles:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True)
        graph = MolecularGraph.from_smiles(smiles)
        graph.dense_edge_index = self.dense_edge_index(graph)
        graph.dense_edge_attr = self.dense_edge_attr(graph)

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
        graph = MolecularGraph.from_smiles(smiles)
        graph.dense_edge_index = self.dense_edge_index(graph)
        graph.dense_edge_attr = self.dense_edge_attr(graph)
        graph.mol_properties = props

        return graph


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
    #atom_feat.extend(one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE_LIST))
    atom_feat.extend(one_hot_encoding(atom.GetNumExplicitHs(), HS_LIST))
    #atom_feat.extend([atom.GetIsAromatic()])
    return atom_feat


def one_hot_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feat = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        #bond.IsInRing()
    ]
    return bond_feat


def rdkit_mol_from_graph(graph):
    graph = graph.to("cpu")
    num_atoms = len(graph.x)
    atom_type = torch.where(graph.x[:, :NUM_ELEMENTS] == 1)[1].tolist()
    atom_type = [ELEM_LIST[idx] for idx in atom_type]
    charge = torch.where(graph.x[:, NUM_ELEMENTS:NUM_ELEMENTS + 5] == 1)[1].tolist()
    charge = [CHARGE_LIST[idx] for idx in charge]
    #hybridization = torch.where(graph.x[:, NUM_ELEMENTS + 5: NUM_ELEMENTS + 5 + 7] == 1)[1].tolist()
    #hybridization = [HYBRIDIZATION_TYPE_LIST[idx] for idx in hybridization]
    num_hs = torch.where(graph.x[:, NUM_ELEMENTS + 5: NUM_ELEMENTS + 5 + 4] == 1)[1].tolist()
    num_hs = [HS_LIST[idx] for idx in num_hs]
    #is_aromatic = graph.x[:, -1].bool().tolist()
    bonds = {}
    bond_type = torch.where(graph.edge_attr[:, :4] == 1)[1].tolist()
    for i in range(graph.edge_index.shape[1]):
        edge_idx = tuple(graph.edge_index[:, i].tolist())
        edge_idx_reversed = (edge_idx[1], edge_idx[0])
        if (edge_idx not in bonds) & (edge_idx_reversed not in bonds):
            bonds[edge_idx] = BOND_LIST[bond_type[i]]
    mol = Chem.RWMol()
    for i in range(num_atoms):
        atom = Chem.Atom(atom_type[i])
        #atom.SetIsAromatic(is_aromatic[i])
        atom.SetFormalCharge(charge[i])
        atom.SetNumExplicitHs(num_hs[i])
        #atom.SetHybridization(hybridization[i])
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


def get_mask_for_batch(batch, device):
    num_elements = torch.bincount(batch)
    max_len = num_elements.max()
    batch_size = len(num_elements)
    mask = torch.where(
        torch.arange(max_len, device=device).unsqueeze(0) < num_elements.unsqueeze(1),
        torch.ones((batch_size, max_len), device=device),
        torch.zeros((batch_size, max_len), device=device)
    )
    mask = mask.bool()
    return mask

