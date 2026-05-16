
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
 
class PreprocessedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_sl':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)
 
class KAN_node_embedding(nn.Module):
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super().__init__()
        self.harmonics = num_harmonics
        self.addbias = addbias
        self.in_size = input_size
        self.out_size = output_size
        self.fouriercoeffs = nn.Parameter(torch.randn(2, output_size, input_size, num_harmonics) / 
                                          (np.sqrt(input_size) * np.sqrt(num_harmonics)))
        k = torch.arange(1, num_harmonics + 1).view(1, 1, 1, num_harmonics)
        self.register_buffer('k', k)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))
 
    def forward(self, x):
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        x_scaled = x_expanded * self.k
        cos_terms = torch.cos(x_scaled)
        sin_terms = torch.sin(x_scaled)
        y_cos = torch.einsum('bnih,oih->bo', cos_terms, self.fouriercoeffs[0])
        y_sin = torch.einsum('bnih,oih->bo', sin_terms, self.fouriercoeffs[1])
        y = y_cos + y_sin
        if self.addbias:
            y = y + self.bias
        return y
 
 
class KAN_message_passing(MessagePassing):
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super().__init__(aggr='add')
        self.KAN = KAN_node_embedding(input_size, output_size, num_harmonics, addbias=False)
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
 
        x = self.KAN(x)
        y = self.propagate(edge_index_sl, x=x, norm=norm)
        if self.addbias:
            y = y + self.bias
        return y
 
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
 
 
class KA_GCN_latent(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_harmonics,
                 num_message_layers, num_readout_layers, use_bias=False):
        super().__init__()
        self.node_embedding = KAN_node_embedding(input_size, hidden_size, num_harmonics, addbias=use_bias)
        self.message_layers = nn.ModuleList([KAN_message_passing(hidden_size, hidden_size, num_harmonics, addbias=use_bias)
                                            for _ in range(num_message_layers)])
        if num_readout_layers == 1:
            layers = [KAN_node_embedding(hidden_size, latent_size, num_harmonics, addbias=use_bias)]
        else:
            layers = [KAN_node_embedding(hidden_size, hidden_size, num_harmonics, addbias=use_bias)
                      for _ in range(num_readout_layers - 1)]
            layers.append(KAN_node_embedding(hidden_size, latent_size, num_harmonics, addbias=use_bias))
        self.latent_readout = nn.Sequential(*layers)
 
    def forward(self, g, features):
        h = self.node_embedding(features)
        ei_sl = getattr(g, 'edge_index_sl', None)
        norm  = getattr(g, 'norm', None)
        for layer in self.message_layers:
            h = layer(h, g.edge_index, precomputed_edge_index=ei_sl, precomputed_norm=norm)
        y   = global_mean_pool(h, g.batch)
        out = self.latent_readout(y)
        return out
 
 
class KA_GAE(nn.Module):
    def __init__(self, in_feat, hidden_feat, latent_feat, out_feat, num_harmonics,
                 e_num_layers, r_num_layers, d_num_layers, use_bias=False):
        super().__init__()
        self.encoder = KA_GCN_latent(in_feat, hidden_feat, latent_feat, num_harmonics,
                                     e_num_layers, r_num_layers, use_bias=use_bias)
        if d_num_layers == 1:
            dec_layers = [KAN_node_embedding(latent_feat, out_feat, num_harmonics, addbias=use_bias)]
        else:
            dec_layers = [KAN_node_embedding(latent_feat, hidden_feat, num_harmonics, addbias=use_bias)]
            for _ in range(d_num_layers - 2):
                dec_layers.append(KAN_node_embedding(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
            dec_layers.append(KAN_node_embedding(hidden_feat, out_feat, num_harmonics, addbias=use_bias))
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
 
def train(model, device, train_loader, valid_loader, optimizer, loss_fn,
          encoding=True, return_auc=False, contrastive_weight=0.1, margin=1.0):
    model.train()
    total_train_loss = 0.0
    for graphs, node_eigvals_target, labels, fps in train_loader:
        optimizer.zero_grad(set_to_none=True)          # faster than zeroing
        graphs = graphs.to(device)
        node_eigvals_target = node_eigvals_target.to(device)
        y = labels.to(device).float()
        fps = fps.to(device)
        if encoding:
            z   = model.encoder(graphs, graphs.x)
            out = model.decoder(z)
            recon = loss_fn(out, node_eigvals_target)
            if contrastive_weight > 0.0:
                tanimoto = batch_tanimoto(fps)
                contrast = contrastive_loss(z, tanimoto, margin)
                loss = recon + contrastive_weight * contrast
            else:
                loss = recon
        else:
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
                all_preds.append(out.view(-1).cpu())
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
 
 
class GraphFeatureDataset(Dataset):
    def __init__(self, graph_list, eigval_list, feat_list, label_list):
        self.graphs = graph_list
        self.target = [torch.cat([eigval_list[i], feat_list[i].view(-1)], dim=0) for i in range(len(eigval_list)) ]
        self.labels = label_list
        self.fps    = [g.fp for g in graph_list]
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, index):
        return self.graphs[index], self.target[index], self.labels[index]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def GAE_KAN_Script(datafile, iterations, learning_rate, enc_epochs, num_harmonics, 
                   num_message_layers, num_readout_layers, num_dec_layers, hidden_width, latent_size, topo_weight=0.5,
                   eval_every=5, patience=50, prediction_model = None, pred_epochs=0, seed=0,
                   contrastive_weight=0.1, margin=1.0):
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GAE_KAN running on GPU: {torch.cuda.get_device_name(0)}')
        n_workers = min(4, os.cpu_count() or 1)
        pin_mem   = True
    else:
        device = torch.device('cpu')
        print('GAE_KAN running on CPU')
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
    pred_loss_fn  = nn.BCELoss()
 
    target_map = {'tox21': 12, 'muv': 17, 'sider': 27,
                  'clintox': 2, 'bace': 1, 'bbbp': 1, 'hiv': 1}
    file_name  = datafile.split("_")[0]
    target_dim = target_map[file_name]
 
    state = torch.load(datafile + '.pth', weights_only=False)
 
    train_graphs = pre_process_graphs(state['train'])
    valid_graphs = pre_process_graphs(state['valid'])
 
    train_graph_targets = pre_process_targets(train_graphs,k=k)
    valid_graph_targets = pre_process_targets(valid_graphs, k=k)
 
    train_gs, train_evs, train_node_feat = zip(*train_graph_targets)
    valid_gs, valid_evs, valid_node_feat = zip(*valid_graph_targets)
 
    train_labels = [g.y for g in state['train']]
    valid_labels = [g.y for g in state['valid']]
 
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
 
    node_dim  = train_graphs[0].x.shape[1]          # inferred, not hardcoded
    out_feat  = k + node_dim                         # eigenvalues + pooled features
  
    All_AUC = []
    for i in range(iterations):
        set_seed(i)
        best_auc = 0.0
        epoch_since_best = 0
        ae_model = KA_GAE(
                in_feat=node_dim, hidden_feat=hidden_width, latent_feat=latent_size,
                out_feat=out_feat, num_harmonics=num_harmonics,
                e_num_layers=num_message_layers, r_num_layers=num_readout_layers,
                d_num_layers=num_dec_layers, use_bias=True,
            ).to(device)
    
        ae_optimiser   = torch.optim.Adam(ae_model.parameters(),    lr=learning_rate)
        #loss_vec,valid_vec=[],[]
        for _ in range(enc_epochs):
            loss,valid,a = train(ae_model, device, train_loader, valid_loader,
                  ae_optimiser, recon_loss_fn, encoding=True, return_auc=False,
                  contrastive_weight=contrastive_weight,margin=margin)
            #loss_vec.append(loss)
            #valid_vec.append(valid)
        
 
        #Prediction must be toggled on
        if prediction_model is not None:
            latent_model = prediction_model
            pred_model = LatentPass(ae_model.encoder, latent_model).to(device)
            pred_optimiser = torch.optim.Adam(latent_model.parameters(), lr=learning_rate) 
            AUC_list = []
            for epoch in range(pred_epochs):
                do_auc = (epoch % eval_every == 0) or (epoch == pred_epochs - 1)
                loss, valid, auc = train(pred_model, device, train_loader, valid_loader,
                                pred_optimiser, pred_loss_fn,
                                encoding=False, return_auc=do_auc,
                                contrastive_weight=0)
                epoch_since_best += 1
                if auc is not None:
                    AUC_list.append(auc)
                    if auc > best_auc:
                        best_auc = auc
                        epoch_since_best = 0
                if epoch_since_best > patience:
                    break
    
            All_AUC.append(best_auc)
    return All_AUC
