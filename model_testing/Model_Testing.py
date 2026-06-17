
import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import scatter
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import random
import torch.nn.functional as F
import time
import optimised_GAEKAN
import optimised_GAEMLP
import json

class PreprocessedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_sl':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)
 
class LatentPass(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, g, x):
        z = self.encoder(g, x)
        return self.predictor(z)
  
 
def train(model, device, train_loader, valid_loader, optimizer, loss_fn,
          encoding=True, return_auc=False):
    model.train()
    total_train_loss = 0.0
    for graphs, node_eigvals_target, labels, fps in train_loader:
        optimizer.zero_grad(set_to_none=True)          # faster than zeroing
        graphs = graphs.to(device)
        node_eigvals_target = node_eigvals_target.to(device)
        y = labels.to(device).float()
        fps = fps.to(device)
        out  = model(graphs, graphs.x)
        loss = loss_fn(out, y)
 
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
 
    model.eval()
    total_loss_val = 0.0
    all_preds  = [] if return_auc else None
    all_labels = [] if return_auc else None
    with torch.no_grad():
        for graphs, node_eigvals_target, labels, fps in valid_loader:
            graphs = graphs.to(device)
            node_eigvals_target = node_eigvals_target.to(device)
            y = labels.to(device).float()
            out = model(graphs, graphs.x)
            loss = loss_fn(out, node_eigvals_target) if encoding else loss_fn(out, y)
            total_loss_val += loss.item()
            if return_auc:
                probs = torch.sigmoid(out)
                all_preds.append(probs.view(-1).cpu())
                all_labels.append(y.view(-1).cpu())
 
    auc = None
    if return_auc:
        preds  = torch.cat(all_preds).numpy()
        labeling = torch.cat(all_labels).numpy()
        auc = roc_auc_score(labeling, preds)
 
    return total_train_loss, total_loss_val, auc
 
def compute_targets_with_graph(g, k=10, method="sum"):
    edge_index, edge_weight = get_laplacian(g.edge_index, normalization=None, num_nodes=g.num_nodes)
    L = to_dense_adj(edge_index, edge_attr=edge_weight,max_num_nodes=g.num_nodes).squeeze(0)
    eigenvalues = torch.linalg.eigvalsh(L)[1:k + 1]  
 
    if method == "sum":
        global_graph_feat = global_add_pool(g.x, g.batch)
    elif method == "avg":
        global_graph_feat = global_mean_pool(g.x, g.batch)
    else:
        raise ValueError("No valid feature selected")
    return g, eigenvalues, global_graph_feat
 
 
def pre_process_targets(graph_list, k=10, feat_method="sum", num_workers=None):
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, len(graph_list))
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(lambda g: compute_targets_with_graph(g, k=k, method=feat_method),graph_list))
    return results
 
 
def pre_process_graphs(graph_list):
    processed = []
    for g in graph_list:
        g = g.clone()
 
        agg_edge_feat = scatter(g.edge_attr, g.edge_index[1], dim=0, dim_size=g.num_nodes, reduce='mean')
        g.x = torch.cat([g.x, agg_edge_feat], dim=1)
 
        edge_index_sl, node = add_self_loops(g.edge_index, num_nodes=g.num_nodes)
        row, col  = edge_index_sl
        deg = degree(col, g.num_nodes, dtype=torch.float)
        d_inv_sq = deg.pow(-0.5)
        d_inv_sq[d_inv_sq == float('inf')] = 0
        norm = d_inv_sq[row] * d_inv_sq[col]
        fp = getattr(g, 'fp', torch.zeros(1024, dtype=torch.float32))
        pd = PreprocessedData(
            x  = g.x,
            edge_index = g.edge_index,
            edge_attr = g.edge_attr,
            y = g.y,
            num_nodes = g.num_nodes,
            edge_index_sl = edge_index_sl,
            norm = norm,
            fp = fp
        )
        processed.append(pd)
    return processed

class MLP_latentpred(nn.Module):
    def __init__(self, latent_feat, hidden_feat, out_feat, p_num_layers, use_bias=True):
        super().__init__()
        if p_num_layers == 1:
            layers = [nn.Linear(latent_feat,out_feat,bias=use_bias)]
        else:
            layers = [nn.Linear(latent_feat,hidden_feat,bias=use_bias),
                      nn.ReLU()]
            for _ in range(p_num_layers - 2):
                layers.append(nn.Linear(hidden_feat,hidden_feat,bias=use_bias))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_feat,out_feat,bias=use_bias))
        self.predictor = nn.Sequential(*layers)
 
    def forward(self, latent):
        return self.predictor(latent) 
 
class GraphFeatureDataset(Dataset):
    def __init__(self, graph_list, eigval_list, feat_list, label_list):
        self.graphs = graph_list
        self.target = [torch.cat([eigval_list[i], feat_list[i].view(-1)], dim=0) for i in range(len(eigval_list)) ]
        self.labels = label_list
        self.fps    = [g.fp for g in graph_list]
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, index):
        return self.graphs[index], self.target[index], self.labels[index], self.fps[index]

class LatentDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = latents   # list of 1-D tensors, already on CPU
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

@torch.no_grad()
def extract_latents(encoder, loader, device):
    encoder.eval()
    latents, labels = [], []
    for graphs, _, label, _ in loader:
        graphs = graphs.to(device)
        z = encoder(graphs, graphs.x)          # (batch, latent_size)
        latents.append(z.cpu())
        labels.append(label)
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)

def train_latent(model, device, train_loader, eval_loader,
                 optimizer, loss_fn, return_auc=False):
    model.train()
    total_train_loss = 0.0
    for z, y in train_loader:
        z, y = z.to(device), y.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        out  = model(z)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_eval_loss = 0.0
    all_preds, all_labels = ([], []) if return_auc else (None, None)
    with torch.no_grad():
        for z, y in eval_loader:
            z, y = z.to(device), y.to(device).float()
            out  = model(z)
            loss = loss_fn(out, y)
            total_eval_loss += loss.item()
            if return_auc:
                all_preds.append(out.view(-1).cpu())
                all_labels.append(y.view(-1).cpu())

    auc = None
    if return_auc:
        auc = roc_auc_score(
            torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy()
        )
    return total_train_loss, total_eval_loss, auc


def Testing_Script(datafile,batch_size, latent_size, saved_model,
                   tune_iter=20,test_iter=100, epochs=1000,
                   eval_every=5, patience=50, seed=0):
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(saved_model, f'running on GPU: {torch.cuda.get_device_name(0)}')
        n_workers = min(4, os.cpu_count() or 1)
        pin_mem   = True
    else:
        device = torch.device('cpu')
        print( saved_model,'running on CPU')
        torch.set_num_threads(os.cpu_count() or 1)
        n_workers = min(8, os.cpu_count() or 1)
        pin_mem   = False
 
    persist = n_workers > 0
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
 
    target_map = {'tox21': 12,'bace': 1, 'bbbp': 1, 'hiv': 1}
    file_name  = datafile.split("_")[0]
    target_dim = target_map[file_name]
 
    state = torch.load(datafile + f'_{batch_size}.pth', weights_only=False)
 
    train_graphs = pre_process_graphs(state['train'])
    valid_graphs = pre_process_graphs(state['valid'])
    test_graphs = pre_process_graphs(state['test'])
 
    train_graph_targets = pre_process_targets(train_graphs)
    valid_graph_targets = pre_process_targets(valid_graphs)
    test_graph_targets = pre_process_targets(test_graphs)
 
    train_gs, train_evs, train_node_feat = zip(*train_graph_targets)
    valid_gs, valid_evs, valid_node_feat = zip(*valid_graph_targets)
    test_gs, test_evs, test_node_feat = zip(*test_graph_targets) 
 
    train_labels = [g.y for g in state['train']]
    valid_labels = [g.y for g in state['valid']]
    test_labels = [g.y for g in state['test']]
 
    loader_bs = int(state['batch_size'])
 
    train_loader = DataLoader(
        GraphFeatureDataset(train_gs, train_evs, train_node_feat, train_labels),
        batch_size=loader_bs, shuffle=state['shuffle'], drop_last=True,
        num_workers=n_workers, persistent_workers=persist, pin_memory=pin_mem
    )
    valid_loader = DataLoader(
        GraphFeatureDataset(valid_gs, valid_evs, valid_node_feat, valid_labels),
        batch_size=loader_bs, shuffle=False, drop_last=True,
        num_workers=n_workers, persistent_workers=persist,  pin_memory=pin_mem
    )
    test_loader = DataLoader(
        GraphFeatureDataset(test_gs, test_evs, test_node_feat, test_labels),
        batch_size=loader_bs, shuffle=False, drop_last=True,
        num_workers=n_workers, persistent_workers=persist,  pin_memory=pin_mem
    )
 
    node_dim  = train_graphs[0].x.shape[1]     

    labels = torch.stack([g.y for g in state['train']])  
    n_pos = labels.sum(dim=0)                             
    n_neg = labels.shape[0] - n_pos                       
    pos_weight = (n_neg / n_pos.clamp(min=1)).to(device)  
    pred_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    #Hyperparameter tuning
    print('Now tuning NN prediction module')
    start = time.perf_counter()

    width = [32,64,128,256]
    depth = [2,3,4]
    lr = [10**-3,10**-4,10**-5]

    model_name = saved_model.split('_')[0]
    tag = saved_model.split('_')[1]
    if tag == 'modified':
        encoder = optimised_GAEKAN.KA_GCN_latent(node_dim,64,128,2,4,3,use_bias=True) #insert my hyperparameters here 
        encoder.load_state_dict(torch.load(saved_model))
        encoder.to(device)
        model_type = 'KAN'
        encoder.eval()
    elif tag == 'mini':
        encoder = optimised_GAEKAN.KA_GCN_latent(node_dim,32,64,1,1,1,use_bias=True) #insert my hyperparameters here 
        encoder.load_state_dict(torch.load(saved_model))
        encoder.to(device)
        model_type = 'KAN'
        encoder.eval()
    elif model_name == 'GAEKAN':
        encoder = optimised_GAEKAN.KA_GCN_latent(node_dim,256,512,2,4,3,use_bias=True) #insert my hyperparameters here 
        encoder.load_state_dict(torch.load(saved_model))
        encoder.to(device)
        model_type = 'KAN'
        encoder.eval()
    elif model_name == 'GAEMLP':
        encoder = optimised_GAEMLP.MLP_GCN_latent(node_dim,64,128,3,3,use_bias=True) #insert my hyperparameters here 
        encoder.load_state_dict(torch.load(saved_model))
        encoder.to(device)
        model_type = 'MLP'
        encoder.eval()      
    else:
        raise ValueError("The file name is wrong, submit a valid file name to load.")  
    for param in encoder.parameters():
        param.requires_grad = False

    print("Pre-computing latent representations...")
    train_z, train_y = extract_latents(encoder, train_loader, device)
    valid_z, valid_y = extract_latents(encoder, valid_loader, device)
    test_z,  test_y  = extract_latents(encoder, test_loader,  device)

    # simple flat DataLoaders — no graph collation needed
    def make_latent_loader(z, y, shuffle=False):
        ds = LatentDataset(list(z), list(y))
        return DataLoader(ds, batch_size=loader_bs, shuffle=shuffle,
                        drop_last=True, num_workers=0)

    latent_train_loader = make_latent_loader(train_z, train_y, shuffle=True)
    latent_valid_loader = make_latent_loader(valid_z, valid_y)
    latent_test_loader  = make_latent_loader(test_z,  test_y)

    best_combo_auc =0
    best_combo = []
    for w in width:
        for d in depth:
            for l in lr:
                all_AUC = []
                print("On model ",w,d,l)
                for j in range(tune_iter):
                    latent_model = MLP_latentpred(latent_size,w,target_dim,d,True).to(device)
                    best_auc = 0.0
                    epoch_since_best = 0
                    pred_optimiser = torch.optim.Adam(latent_model.parameters(), lr=l)
                    AUC_list=[]

                    for epoch in range(epochs):
                        do_auc = (epoch % eval_every == 0) or (epoch == epochs - 1)
                        loss, valid_loss, auc = train_latent(
                            latent_model, device,
                            latent_train_loader, latent_valid_loader,
                            pred_optimiser, pred_loss_fn,
                            return_auc=do_auc
                        )
                        epoch_since_best += 1
                        if auc is not None:
                            AUC_list.append(auc)
                            if auc > best_auc:
                                best_auc = auc
                                epoch_since_best = 0
                        if epoch_since_best > patience:
                            break
                    all_AUC.append(best_auc)
                if np.mean(all_AUC)>best_combo_auc:
                    best_combo_auc = np.mean(all_AUC)
                    best_combo = [w,d,l]
                print('This auc -', np.mean(all_AUC))
    end = time.perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f'Best combo was - width={best_combo[0]}, depth={best_combo[1]},lr={best_combo[2]}')
    print(f'Best auc was ',best_combo_auc)
    best_dict = {'width':best_combo[0], "depth":best_combo[1],"lr":best_combo[2]}
    name = model_type + datafile + 'best_pred_params'
    with open(name, 'w') as f:
        json.dump(best_dict, f, indent=4)

    start = time.perf_counter()
    all_AUC = []
    print('Now running on test set...')
    for j in range(test_iter):
        set_seed(j)
        latent_model= MLP_latentpred(latent_size,best_combo[0],target_dim,best_combo[1],True).to(device)
        best_auc = 0.0
        epoch_since_best = 0
        pred_optimiser = torch.optim.Adam(latent_model.parameters(), lr=best_combo[2])
        AUC_list = []
        for epoch in range(epochs):
            do_auc = (epoch % eval_every == 0) or (epoch == epochs - 1)
            loss, valid_loss, auc = train_latent(
                latent_model, device,
                latent_train_loader, latent_test_loader,
                pred_optimiser, pred_loss_fn,
                return_auc=do_auc
            )
            epoch_since_best += 1
            if auc is not None:
                AUC_list.append(auc)
                if auc > best_auc:
                    best_auc = auc
                    epoch_since_best = 0
            if epoch_since_best > patience:
                break
        all_AUC.append(best_auc)
    end = time.perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")
    print('All done!!')
    return(all_AUC)
