import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import random
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy

from networkx.generators.random_graphs import *
from networkx.generators.ego import ego_graph
from networkx.generators.geometric import random_geometric_graph



class GeometricGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = random_geometric_graph(n=n, radius=0.5)
        return g


class RegularGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = random_regular_graph(n=n, d=4)
        return g


class BarabasiAlbertGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, m_min=1, m_max=5,
                 samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.n_min == self.n_max:
            n = self.m_min
        else:
            n = np.random.randint(low=self.n_min, high=self.n_max)
        if self.m_min == self.m_max:
            m = self.m_min
        else:
            m = np.random.randint(low=self.m_min, high=self.m_max)
        g = barabasi_albert_graph(n, m)
        return g


class BinomialGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, p_min=0.4, p_max=0.6,
                 samples_per_epoch=100000, pyg=False, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min
        self.p_max = p_max
        self.samples_per_epoch = samples_per_epoch
        self.pyg = pyg

    def __len__(self):
        return self.samples_per_epoch

    def get_largest_subgraph(self, g):
        g = g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])
        g = nx.convert_node_labels_to_integers(g, first_label=0)
        return g

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        if self.p_min == self.p_max:
            p = self.p_min
        else:
            p = np.random.randint(low=self.p_min, high=self.p_max)
        p = np.random.uniform(low=self.p_min, high=self.p_max)
        g = binomial_graph(n, p)
        if self.pyg:
            g = from_networkx(g)
        return g

class RandomGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000, **kwargs):
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
    def from_sparse_graph_list(cls, data_list, labels=False):
        if labels:
            max_num_nodes = max([graph.number_of_nodes() for graph, label in data_list])
        else:
            max_num_nodes = max([graph.number_of_nodes() for graph in data_list])
        node_features = []
        edge_features = []
        mask = []
        y = []
        props = []
        for data in data_list:
            if labels:
                graph, label = data
                y.append(label)
            else:
                graph = data
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            nf = torch.ones(max_num_nodes, 1)
            node_features.append(nf.unsqueeze(0))
            dm = torch.from_numpy(floyd_warshall_numpy(graph)).long()
            dm = torch.clamp(dm, 0, 5).unsqueeze(-1)
            num_nodes = dm.size(1)
            dm = torch.zeros((num_nodes, num_nodes, 6)).type_as(dm).scatter_(2, dm, 1).float()
            edge_features.append(dm)
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
        node_features = torch.cat(node_features, dim=0)
        edge_features = torch.stack(edge_features, dim=0)
        mask = torch.cat(mask, dim=0)
        props = torch.cat(props, dim=0)
        batch = cls(node_features=node_features, edge_features=edge_features, mask=mask, properties=props)
        if labels:
            batch.y = torch.Tensor(y)
        return batch

    def __repr__(self):
        repr_list = ["{}={}".format(key, list(value.shape)) for key, value in self.__dict__.items()]
        return "DenseGraphBatch({})".format(", ".join(repr_list))


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, labels=False, **kwargs):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=lambda data_list: DenseGraphBatch.from_sparse_graph_list(data_list, labels), **kwargs)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, graph_family, graph_kwargs=None, samples_per_epoch=100000, batch_size=32,
                 distributed_sampler=True, num_workers=1):
        super().__init__()
        if graph_kwargs is None:
            graph_kwargs = {}
        self.graph_family = graph_family
        self.graph_kwargs = graph_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.distributed_sampler = distributed_sampler
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None

    def make_dataset(self, samples_per_epoch):
        if self.graph_family == "binomial":
            ds = BinomialGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "barabasi_albert":
            ds = BarabasiAlbertGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "regular":
            ds = RegularGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == "geometric":
            ds = GeometricGraphDataset(samples_per_epoch=samples_per_epoch)
        elif self.graph_family == "all":
            ds = RandomGraphDataset(samples_per_epoch=samples_per_epoch)
        else:
            raise NotImplementedError
        return ds

    def train_dataloader(self):
        self.train_dataset = self.make_dataset(samples_per_epoch=self.samples_per_epoch)
        if self.distributed_sampler:
            train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                shuffle=False
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
        self.eval_dataset = self.make_dataset(samples_per_epoch=4096)
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
        if self.pyg:
            g = from_networkx(graph)
            if g.pos is not None:
                del g.pos
            if g.edge_index.dtype != torch.long:
                print(g)
            g.y = torch.Tensor([label]).long()
            return g
        else:
            return graph, label



class EvalRandomBinomialGraphDataset(Dataset):
    def __init__(self, n_min, n_max, p_min, p_max, num_samples, pyg=False):
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min
        self.p_max = p_max
        self.num_samples = num_samples
        self.pyg = pyg
        self.graphs, self.labels = self.generate_dataset()

    def generate_dataset(self):
        graphs = []
        labels = []
        for i in range(self.num_samples):
            n = np.random.randint(low=self.n_min, high=self.n_max)
            p = np.random.uniform(low=self.p_min, high=self.p_max)
            g = binomial_graph(n, p)
            if self.pyg:
                g = from_networkx(g)
                g.y = p
            graphs.append(g)
            labels.append(p)
        return graphs, labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.pyg:
            return graph
        else:
            label = self.labels[idx]
            return graph, label
