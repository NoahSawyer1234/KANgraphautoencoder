
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
 
class PreprocessedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_sl':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs) 
 
class MLP_message_passing(MessagePassing):
    def __init__(self, input_size, output_size, addbias=True):
        super().__init__(aggr='add')
        self.lin = nn.Linear(input_size,output_size, bias=False)
        self.addbias = addbias
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))
 
    def forward(self, x, edge_index, precomputed_edge_index=None, precomputed_norm=None):
        if precomputed_edge_index is not None and precomputed_norm is not None:
            edge_index_sl = precomputed_edge_index
            norm = precomputed_norm
        else:
            edge_index_sl, node = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index_sl
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
 
        x = self.lin(x)
        y = self.propagate(edge_index_sl, x=x, norm=norm)
        if self.addbias:
            y = y + self.bias
        return y
 
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

 
class MLP_GCN_latent(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size,
                 num_message_layers, num_readout_layers, use_bias=False):
        super().__init__()
        self.node_embedding = nn.Linear(input_size, hidden_size, use_bias)
        self.message_layers = nn.ModuleList([MLP_message_passing(hidden_size, hidden_size, addbias=use_bias)
                                            for _ in range(num_message_layers)])
        if num_readout_layers == 1:
            layers = nn.Linear(hidden_size,latent_size,bias=use_bias)
        else:
            layers = [nn.Linear(hidden_size,hidden_size,bias=use_bias),
                      nn.LeakyReLU()]*(num_readout_layers - 1)
            layers.append(nn.Linear(hidden_size, latent_size, bias=use_bias))
        self.latent_readout = nn.Sequential(*layers)
 
    def forward(self, g, features):
        h = self.node_embedding(features)
        ei_sl = getattr(g, 'edge_index_sl', None)
        norm  = getattr(g, 'norm', None)
        for layer in self.message_layers:
            h = layer(h, g.edge_index, precomputed_edge_index=ei_sl, precomputed_norm=norm)
        y = global_mean_pool(h, g.batch)
        out = self.latent_readout(y)
        return out
 
 
class MLP_GAE(nn.Module):
    def __init__(self, in_feat, hidden_feat, latent_feat, out_feat,
                 e_num_layers, r_num_layers, d_num_layers, use_bias=False):
        super().__init__()
        self.encoder = MLP_GCN_latent(in_feat, hidden_feat, latent_feat,
                                     e_num_layers, r_num_layers, use_bias=use_bias)
        #Should never be used
        if d_num_layers == 1:
            dec_layers = [nn.Linear(latent_feat, out_feat, bias=use_bias)]
        else:
            dec_layers = [nn.Linear(latent_feat,hidden_feat,bias=use_bias),
                      nn.LeakyReLU()]
            for _ in range(d_num_layers - 2):
                dec_layers.append(nn.Linear(hidden_feat,hidden_feat,bias=use_bias))
                dec_layers.append(nn.LeakyReLU())
            dec_layers.append(nn.Linear(hidden_feat,out_feat,bias=use_bias))
        self.decoder = nn.Sequential(*dec_layers)
 
    def forward(self, g, features):
        z   = self.encoder(g, features)
        out = self.decoder(z)
        return out
 
 
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
    
def batch_tanimoto(fps):
    dot   = fps @ fps.T                              # [B, B]
    norms = fps.sum(dim=1, keepdim=True)             # [B, 1]  (number of set bits)
    denom = norms + norms.T - dot                    # [B, B]
    denom = denom.clamp(min=1e-8)                    # avoid division by zero
    return dot / denom                               # [B, B]
 
 
def contrastive_loss(z, tanimoto_sim, margin=1.0):
    diff = z.unsqueeze(0) - z.unsqueeze(1)               # [B, B, latent_dim]
    dist = diff.pow(2).sum(dim=2).clamp(min=1e-8).sqrt() # [B, B]
 
    similar_loss    = tanimoto_sim * dist.pow(2)
    dissimilar_loss = (1 - tanimoto_sim) * F.relu(margin - dist).pow(2)

    mask = 1 - torch.eye(z.size(0), device=z.device)
    loss = (similar_loss + dissimilar_loss) * mask
    return loss.sum() / mask.sum()
 
def train(model, device, train_loader, optimizer, loss_fn, contrastive_weight=0.1, margin=1.0):
    model.train()
    total_train_loss = 0.0
    for graphs, node_eigvals_target, labels, fps in train_loader:
        optimizer.zero_grad(set_to_none=True)          # faster than zeroing
        graphs = graphs.to(device)
        node_eigvals_target = node_eigvals_target.to(device)
        y = labels.to(device).float()
        fps = fps.to(device)
        z   = model.encoder(graphs, graphs.x)
        out = model.decoder(z)
        recon = loss_fn(out, node_eigvals_target)
        if contrastive_weight > 0.0:
            tanimoto = batch_tanimoto(fps)
            contrast = contrastive_loss(z, tanimoto, margin)
            loss = recon + contrastive_weight * contrast
        else:
            loss = recon
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss
 
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def GAEMLP_latent_train(datafile, learning_rate, enc_epochs, 
                   num_message_layers, num_readout_layers, num_dec_layers, hidden_width, latent_size, topo_weight=0.5, 
                   seed=0, contrastive_weight=0.1, margin=1.0):
    set_seed(seed)
    start = time.perf_counter()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GAE_MLP running on GPU: {torch.cuda.get_device_name(0)}')
        n_workers = min(4, os.cpu_count() or 1)
        pin_mem   = True
    else:
        device = torch.device('cpu')
        print('GAE_MLP running on CPU')
        torch.set_num_threads(os.cpu_count() or 1)
        n_workers = min(8, os.cpu_count() or 1)
        pin_mem   = False
 
    persist = n_workers > 0
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    k = 10
    _l1 = nn.L1Loss()
    recon_loss_fn = lambda pred, target: (
        _l1(pred[:, :k], target[:, :k])*topo_weight +
        _l1(pred[:, k:], target[:, k:])*(1-topo_weight)) 
 
    state = torch.load(datafile + '.pth', weights_only=False)
 
    train_graphs = pre_process_graphs(state['train'])
    train_graph_targets = pre_process_targets(train_graphs,k=k)
    train_gs, train_evs, train_node_feat = zip(*train_graph_targets)
    train_labels = [g.y for g in state['train']]
    loader_bs = int(state['batch_size'])
    train_loader = DataLoader(
        GraphFeatureDataset(train_gs, train_evs, train_node_feat, train_labels),
        batch_size=loader_bs, shuffle=state['shuffle'], drop_last=True,
        num_workers=n_workers, persistent_workers=persist, pin_memory=pin_mem
    )
    node_dim  = train_graphs[0].x.shape[1]      
    out_feat  = k + node_dim                        
    ae_model = MLP_GAE(
            in_feat=node_dim, hidden_feat=hidden_width, latent_feat=latent_size,
            out_feat=out_feat,
            e_num_layers=num_message_layers, r_num_layers=num_readout_layers,
            d_num_layers=num_dec_layers, use_bias=True,
        ).to(device)

    ae_optimiser   = torch.optim.Adam(ae_model.parameters(), lr=learning_rate)
    for i in range(enc_epochs):
        if i != 0:
            start = time.perf_counter()
        loss = train(ae_model, device, train_loader,
                ae_optimiser, recon_loss_fn,
                contrastive_weight=contrastive_weight,
                margin=margin)
        end = time.perf_counter()
        print("GAEKAN - Finished Epoch ",i, f"took {start-end}")
    torch.save(ae_model.encoder.state_dict(), 'GAEMLP_trained_encoder.pth')

if __name__ == '__main__':
    enc_epochs = 10
    GAEMLP_latent_train('small_molecules_512',0.001,enc_epochs,3,3,2,64,128,
                        topo_weight=0.5,seed=0,contrastive_weight=0.5,margin=1.0) #Manually put my tuned variables


