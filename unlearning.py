# unlearning.py
# This is the pipline where any unlearning method can fit in. 
import torch
from torch_geometric.datasets import CitationFull, Coauthor,Planetoid
from models import GCN
from torch_geometric.utils import subgraph
from torch_geometric.loader import NeighborLoader
import time
def split_data(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_cutoff = int(train_ratio * num_nodes)
    val_cutoff = int((train_ratio + val_ratio) * num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[:train_cutoff]] = True
    data.val_mask[indices[train_cutoff:val_cutoff]] = True
    data.test_mask[indices[val_cutoff:]] = True
    return data
def prepare_pre_unlearning():
    #dataset = CitationFull(root='.', name="citeseer")
    dataset = Coauthor(root='.', name="physics")
    data = dataset[0]
    data = split_data(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = GCN(data.num_node_features, 64, dataset.num_classes).to(device)
    train_loader = NeighborLoader(data, num_neighbors=[5, 10], batch_size=64, input_nodes=data.train_mask)
    val_loader = NeighborLoader(data, num_neighbors=[5, 10], batch_size=64, input_nodes=data.val_mask)
    model.fit(data, 10, train_loader, val_loader)
    forget_nodes = torch.randperm(data.num_nodes)[:int(0.05 * data.num_nodes)]
    return model, data, forget_nodes

def retrain_from_scratch(model, data, forget_nodes):
    print("\n[During Unlearning] Full model reinitialization and retraining...")
    device = data.edge_index.device
    remaining_nodes = torch.tensor([n for n in range(data.num_nodes) if n not in forget_nodes], device=device)
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
    mapped_forget_nodes = [node_map[n.item()] for n in forget_nodes if n.item() in node_map]
    mapped_forget_nodes = torch.tensor(mapped_forget_nodes, dtype=torch.long, device=device)
    sub_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
    sub_data = data.__class__(
        x=data.x[remaining_nodes],
        edge_index=sub_edge_index,
        y=data.y[remaining_nodes]
    ).to(device)
    sub_data = split_data(sub_data)
    model = GCN(sub_data.num_node_features, 64, torch.max(sub_data.y).item() + 1).to(device)
    train_loader = NeighborLoader(sub_data, num_neighbors=[5, 10], batch_size=64, input_nodes=sub_data.train_mask)
    val_loader = NeighborLoader(sub_data, num_neighbors=[5, 10], batch_size=64, input_nodes=sub_data.val_mask)
    import time
    start = time.time()
    model.fit(sub_data, 100, train_loader, val_loader) 
    end = time.time()
    return model, sub_data, end - start, mapped_forget_nodes
def graph_delete(model, data, forget_nodes): # put logic of GNNDelete
    return model, sub_data, end - start, torch.tensor(mapped, dtype=torch.long, device=device)

def idea_method(model, data, forget_nodes): # put logic for IDEA

    return model, data, end - start, forget_nodes

def graph_editor(model, data, forget_nodes): # put logic for GraphEditor
   
    return model, data, end - start, forget_nodes


def perform_unlearning(strategy="idea", model=None, data=None, forget_nodes=None):
    if strategy == "retrain":
        return retrain_from_scratch(model, data, forget_nodes)
    elif strategy == "grapheditor":
        return graph_editor(model, data, forget_nodes)
    elif strategy == "gnndelete":
        return graph_delete(model, data, forget_nodes)
    elif strategy == "idea":
        return idea_method(model, data, forget_nodes)
    else:
        raise ValueError(f"Unknown unlearning strategy: {strategy}")







