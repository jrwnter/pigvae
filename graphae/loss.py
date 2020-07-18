import torch
from torch.nn.functional import mse_loss


def resemblance_loss(mol_emb_real, mol_emb_pred):
    loss = mse_loss(
        input=mol_emb_real,
        target=mol_emb_pred
    )
    return loss


def divergence_loss(mol_emb):
    loss = mse_loss(
        input=mol_emb,
        target=mol_emb[torch.arange(len(mol_emb) - 1, -1, -1).type_as(mol_emb).long()]
    )
    loss = 1 / loss
    return loss


def node_count_loss(mask, mask_gen):
    mask = mask > 0.5
    num_nodes = torch.sum(mask, dim=1).float()
    sorted_node_prob = torch.sort(mask_gen, descending=True)[0]
    top_nodes_sum = torch.sum(sorted_node_prob * mask, dim=1)
    rest_node_sum = torch.sum(sorted_node_prob * ~mask, dim=1)
    loss = mse_loss(
        input=top_nodes_sum - rest_node_sum,
        target=num_nodes
    )
    return loss

def kld_loss(mu, logvar):
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
    loss = torch.mean(loss)
    return loss


def edge_count_loss(adj, adj_gen):
    batch_size, max_num_nodes, _ = adj.shape
    num_edges = (adj.triu(1) > 0.5).sum(dim=(1, 2)).float()
    mask = (torch.arange(max_num_nodes * max_num_nodes).type_as(adj).view(1, -1).expand(batch_size, -1) < num_edges.unsqueeze(1))
    sorted_edge_prob = torch.sort(adj_gen.triu(1).flatten(start_dim=1), descending=True)[0]
    top_edges_sum = torch.sum(sorted_edge_prob * mask, dim=1)
    rest_edges_sum = torch.sum(sorted_edge_prob * ~mask, dim=1)
    loss = mse_loss(
        input=top_edges_sum - rest_edges_sum,
        target=num_edges
    )
    return loss


def critic(mol_emb, mol_emb_gen, mask, mask_gen, adj, adj_gen):
    loss = {
        "resemblance_loss": resemblance_loss(mol_emb, mol_emb_gen),
        "divergence_loss": divergence_loss(mol_emb),
        "node_count_loss": node_count_loss(mask, mask_gen),
        "edge_count_loss": edge_count_loss(adj, adj_gen),
    }
    loss["total_loss"] = sum(loss.values())
    return loss
