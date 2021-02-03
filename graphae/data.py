import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem.rdmolops import GetDistanceMatrix
from torch_geometric.data import Data
from torch_geometric.transforms import ToDense
from torch_geometric.utils import from_networkx
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

from networkx.generators.random_graphs import *
from networkx.generators.ego import ego_graph
from networkx.generators.geometric import random_geometric_graph


MEAN_DISTANCE = 2.0626
STD_DISTANCE = 1.1746

NODE_FEATURES = torch.eye(20).unsqueeze(0)


class RandomGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch
        self.graph_generator = GraphGenerator()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = self.graph_generator(n)
        return g


class PyGRandomGraphDataset(RandomGraphDataset):
    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = self.graph_generator(n)
        g = from_networkx(g)
        if g.pos is not None:
            del g.pos
        return g



class DenseGraphBatch(Data):
    def __init__(self, node_features, edge_features, mask, **kwargs):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        for key, item in kwargs.items():
            setattr(self, key, item)

    @classmethod
    def from_sparse_graph_list(cls, graph_list):
        max_num_nodes = max([graph.number_of_nodes() for graph in graph_list])
        node_features = []
        edge_features = []
        mask = []
        for graph in graph_list:
            num_nodes = graph.number_of_nodes()
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            nf = torch.ones(max_num_nodes, 1)
            node_features.append(nf.unsqueeze(0))
            adj = torch.from_numpy(np.array(adjacency_matrix(graph).todense())).float().unsqueeze(-1)
            edge_features.append(adj.clone())
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
        node_features = torch.cat(node_features, dim=0)
        edge_features = torch.stack(edge_features, dim=0)
        mask = torch.cat(mask, dim=0)
        return cls(node_features=node_features, edge_features=edge_features, mask=mask)

    def __repr__(self):
        repr_list = ["{}={}".format(key, list(value.shape)) for key, value in self.__dict__.items()]
        return "DenseGraphBatch({})".format(", ".join(repr_list))


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=lambda data_list: DenseGraphBatch.from_sparse_graph_list(data_list), **kwargs)


class MolecularGraphDataModule(pl.LightningDataModule):
    def __init__(self, graph_kwargs, samples_per_epoch=100000, batch_size=32, num_workers=1):
        super().__init__()
        self.graph_kwargs = graph_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None

    def train_dataloader(self):
        train_dataset = RandomGraphDataset(samples_per_epoch=self.samples_per_epoch, **self.graph_kwargs)
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            shuffle=False
        )
        return DenseGraphDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        eval_dataset = RandomGraphDataset(samples_per_epoch=8192, **self.graph_kwargs)
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


def binomial_ego_graph(n, p):
    g = ego_graph(binomial_graph(n, p), 0)
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    return g


class GraphGenerator(object):
    def __init__(self):
        self.graph_params = {
            "binominal": {
                "func": binomial_graph,
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },
            """"binominal_ego": {
                "func": binomial_ego_graph,
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },"""
            "newman_watts_strogatz": {
                "func": newman_watts_strogatz_graph,
                "kwargs_int_ranges": {
                    "k": (2, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },
            "watts_strogatz": {
                "func": watts_strogatz_graph,
                "kwargs_int_ranges": {
                    "k": (2, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.2, 0.6)
                }
            },
            "random_regular": {
                "func": random_regular_graph,
                "kwargs_int_ranges": {
                    "d": (3, 6),  # n*d must be even
                }
            },
            "barabasi_albert": {
                "func": barabasi_albert_graph,
                "kwargs_int_ranges": {
                    "m": (1, 6),
                }
            },
            "dual_barabasi_albert": {
                "func": dual_barabasi_albert_graph,
                "kwargs_int_ranges": {
                    "m1": (1, 6),
                    "m2": (1, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.1, 0.9)
                }
            },
            "extended_barabasi_albert": {
                "func": extended_barabasi_albert_graph,
                "kwargs_int_ranges": {
                    "m": (1, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.1, 0.49),
                    "q": (0.1, 0.49)
                }

            },
            "powerlaw_cluster": {
                "func": powerlaw_cluster_graph,
                "kwargs_int_ranges": {
                    "m": (1, 6),
                },
                "kwargs_float_ranges": {
                    "p": (0.1, 0.9),
                }
            },
            "random_powerlaw_tree": {
                "func": random_powerlaw_tree,
                "kwargs": {
                    "gamma": 3,
                    "tries": 1000
                }
            },
            "random_geometric": {
                "func": random_geometric_graph,
                "kwargs_float_ranges": {
                    "p": (0.4, 0.5),
                },
                "kwargs": {
                    "radius": 1
                }
            }
        }
        self.graph_types = list(self.graph_params.keys())

    def __call__(self, n, graph_type=None):
        if graph_type is None:
            graph_type = random.choice(self.graph_types)
        params = self.graph_params[graph_type]
        kwargs = {}
        if "kwargs" in params:
            kwargs = {**params["kwargs"]}
        if "kwargs_int_ranges" in params:
            for key, arg in params["kwargs_int_ranges"].items():
                kwargs[key] = np.random.randint(arg[0], arg[1] + 1)
        if "kwargs_float_ranges" in params:
            for key, arg in params["kwargs_float_ranges"].items():
                kwargs[key] = np.random.uniform(arg[0], arg[1])

        # check if d * n even
        if graph_type == "random_regular":
            if n * kwargs["d"] % 2 != 0:
                n -= 1
        try:
            g = params["func"](n=n, **kwargs)
        except nx.exception.NetworkXError:
            g = self(n)
        return g


class EvalRandomGraphDataset(Dataset):
    def __init__(self, n, pyg=False):
        self.n = n
        self.pyg = pyg
        self.graph_params = {
            "binominal": {
                "func": binomial_graph,
                "kwargs": {
                    "p": (0.25, 0.35, 0.5)
                }
            },
            "binominal_ego": {
                "func": binomial_ego_graph,
                "kwargs": {
                    "p": (0.25, 0.5)
                }
            },
            "newman_watts_strogatz": {
                "func": newman_watts_strogatz_graph,
                "kwargs": {
                    "k": (2, 2, 5, 5),
                    "p": (0.25, 0.75, 0.25, 0.75,)
                }
            },
            "watts_strogatz": {
                "func": watts_strogatz_graph,
                "kwargs": {
                    "k": (2, 2, 5, 5),
                    "p": (0.25, 0.75, 0.25, 0.75,)
                }
            },
            "random_regular": {
                "func": random_regular_graph,
                "kwargs": {
                    "d": (3, 4, 5, 6)
                }
            },
            "barabasi_albert": {
                "func": barabasi_albert_graph,
                "kwargs": {
                    "m": (1, 2, 3, 4),
                }
            },
            "dual_barabasi_albert": {
                "func": dual_barabasi_albert_graph,
                "kwargs": {
                    "m1": (2, 2),
                    "m2": (4, 1),
                    "p": (0.5, 0.5)
                }
            },
            "extended_barabasi_albert": {
                "func": extended_barabasi_albert_graph,
                "kwargs": {
                    "m": (1, 2, 4),
                    "p": (0.5, 0.5, 0.5),
                    "q": (0.25, 0.25, 0.25)
                }

            },
            "powerlaw_cluster": {
                "func": powerlaw_cluster_graph,
                "kwargs": {
                    "m": (2, 3, 4),
                },
                "kwargs_fix": {
                    "p": 0.5
                }
            },
            "random_powerlaw_tree": {
                "func": random_powerlaw_tree,
                "kwargs_fix": {
                    "gamma": 3,
                    "tries": 10000
                }
            },
            "random_geometric": {
                "func": random_geometric_graph,
                "kwargs": {
                    "p": (0.35, 0.55),
                },
                "kwargs_fix": {
                    "radius": 1
                }
            }
        }
        # no ego
        self.graph_types = ["binominal", "barabasi_albert", "random_geometric", "random_regular",
                            "random_powerlaw_tree", "watts_strogatz", "extended_barabasi_albert",
                       "newman_watts_strogatz", "dual_barabasi_albert"]
        graphs, labels = self.generate_dataset()
        c = list(zip(graphs, labels))

        random.shuffle(c)

        self.graphs, self.labels = zip(*c)

    def generate_dataset(self):
        label = 0
        graphs = []
        labels = []
        for j, graph_type in enumerate(self.graph_types):
            params = self.graph_params[graph_type]
            func = params["func"]
            if "kwargs" in params:
                kwargs = params["kwargs"]
            else:
                kwargs = None
            if "kwargs_fix" in params:
                kwargs_fix = params["kwargs_fix"]
            else:
                kwargs_fix = None
            if kwargs is not None:
                num_settings = len(list(kwargs.values())[0])
            else:
                num_settings = 1
            for i in range(num_settings):
                final_kwargs = {}
                if kwargs is not None:
                    for key, args in kwargs.items():
                        if num_settings > 1:
                            final_kwargs[key] = args[i]
                        else:
                            final_kwargs[key] = args
                num_graphs = int(256 / num_settings)
                if kwargs_fix is not None:
                    final_kwargs2 = {**final_kwargs, **kwargs_fix}
                elif kwargs is None:
                    final_kwargs2 = kwargs_fix
                else:
                    final_kwargs2 = final_kwargs
                gs = [func(n=self.n, **final_kwargs2) for _ in range(num_graphs)]
                graphs.extend(gs)
                labels.extend(len(gs) * [label])
                label += 1
        return graphs, labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        #print(graph.number_of_edges())
        if self.pyg:
            g = from_networkx(graph)
            if g.pos is not None:
                del g.pos
            if g.edge_index.dtype != torch.long:
                print(g)
            #g = Data(edge_index=g.edge_index, num_nodes=graph.number_of_nodes)
            g.y = torch.Tensor([label]).long()
            return g
        else:
            return graph, label
