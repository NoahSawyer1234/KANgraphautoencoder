# GAE_KAN_GPU.py  —  CPU + GPU optimised version
#
# Builds on all CPU optimisations and adds:
#   1. Automatic Mixed Precision (AMP)  – runs forward/backward in float16 on
#      GPU via torch.amp.autocast + GradScaler, typically 1.5–3× faster.
#   2. pin_memory=True in DataLoaders  – pages data into pinned (page-locked)
#      host memory for faster CPU→GPU transfers.
#   3. non_blocking=True on .to(device)  – overlaps data transfer with compute.
#   4. torch.backends.cudnn.benchmark=True  – lets cuDNN auto-tune kernels for
#      your specific input sizes (disable if you need strict reproducibility).
#   5. Falls back to CPU automatically when CUDA is unavailable (AMP is a
#      no-op on CPU so the same code path is safe in both cases).
 
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
from torch_geometric.data.data import DataEdgeAttr
from concurrent.futures import ThreadPoolExecutor
from torch.amp import autocast, GradScaler
 
torch.serialization.add_safe_globals([DataEdgeAttr])
 
 
# ---------------------------------------------------------------------------
# Custom Data subclass so edge_index_sl is correctly offset when batching
# ---------------------------------------------------------------------------
class PreprocessedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_sl':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)
 
 
# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
 
class KAN_node_embedding(nn.Module):
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super().__init__()
        self.harmonics = num_harmonics
        self.addbias   = addbias
        self.in_size   = input_size
        self.out_size  = output_size
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, output_size, input_size, num_harmonics) /
            (np.sqrt(input_size) * np.sqrt(num_harmonics))
        )
        k = torch.arange(1, num_harmonics + 1).view(1, 1, 1, num_harmonics)
        self.register_buffer('k', k)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))
 
    def forward(self, x):
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        x_scaled   = x_expanded * self.k
        cos_terms  = torch.cos(x_scaled)
        sin_terms  = torch.sin(x_scaled)
        y_cos = torch.einsum('bnih,oih->bo', cos_terms, self.fouriercoeffs[0])
        y_sin = torch.einsum('bnih,oih->bo', sin_terms, self.fouriercoeffs[1])
        y = y_cos + y_sin
        if self.addbias:
            y = y + self.bias
        return y
 
 
class KAN_message_passing(MessagePassing):
    """GCN-style message passing with KAN transformation.
 
    Uses precomputed edge_index_sl / norm if available on the batched graph.
    """
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super().__init__(aggr='add')
        self.KAN     = KAN_node_embedding(input_size, output_size, num_harmonics, addbias=False)
        self.addbias = addbias
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))
 
    def forward(self, x, edge_index, precomputed_edge_index=None, precomputed_norm=None):
        if precomputed_edge_index is not None and precomputed_norm is not None:
            edge_index_sl = precomputed_edge_index
            norm          = precomputed_norm
        else:
            edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index_sl
            deg      = degree(col, x.size(0), dtype=x.dtype)
            d_inv_sq = deg.pow(-0.5)
            d_inv_sq[d_inv_sq == float('inf')] = 0
            norm = d_inv_sq[row] * d_inv_sq[col]
 
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
        self.message_layers = nn.ModuleList([
            KAN_message_passing(hidden_size, hidden_size, num_harmonics, addbias=use_bias)
            for _ in range(num_message_layers)
        ])
        if num_readout_layers == 1:
            layers = [KAN_node_embedding(hidden_size, latent_size, num_harmonics, addbias=use_bias)]
        else:
            layers = [KAN_node_embedding(hidden_size, hidden_size, num_harmonics, addbias=use_bias)
                      for _ in range(num_readout_layers - 1)]
            layers.append(KAN_node_embedding(hidden_size, latent_size, num_harmonics, addbias=use_bias))
        self.latent_readout = nn.Sequential(*layers)
 
    def forward(self, g, features):
        h     = self.node_embedding(features)
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
 
 
class KA_latentpred(nn.Module):
    def __init__(self, latent_feat, hidden_feat, out_feat, num_harmonics, p_num_layers, use_bias=True):
        super().__init__()
        if p_num_layers == 1:
            layers = [KAN_node_embedding(latent_feat, out_feat, num_harmonics, addbias=use_bias)]
        else:
            layers = [KAN_node_embedding(latent_feat, hidden_feat, num_harmonics, addbias=use_bias)]
            for _ in range(p_num_layers - 2):
                layers.append(KAN_node_embedding(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
            layers.append(KAN_node_embedding(hidden_feat, out_feat, num_harmonics, addbias=use_bias))
        layers.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*layers)
 
    def forward(self, latent):
        return self.predictor(latent)
 
 
class LatentPass(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder   = encoder
        self.predictor = predictor
        for param in self.encoder.parameters():
            param.requires_grad = False
 
    def forward(self, g, x):
        z = self.encoder(g, x)
        return self.predictor(z)
 
 
# ---------------------------------------------------------------------------
# Training utilities  (GPU version with AMP)
# ---------------------------------------------------------------------------
 
def train(model, device, train_loader, valid_loader, optimizer, loss_fn,
          encoding=True, return_auc=False, scaler=None):
    """Single epoch: training then validation in one pass.
 
    scaler : GradScaler instance (pass None to disable AMP, e.g. on CPU).
    return_auc : if True, AUC is computed during the validation loop so no
                 separate predicting() pass is needed.
    """
    use_amp = (scaler is not None) and (device.type == 'cuda')
    amp_ctx = autocast('cuda') if use_amp else torch.amp.autocast('cpu', enabled=False)
 
    # ---- training ----
    model.train()
    total_train_loss = 0.0
    for graphs, node_eigvals_target, labels in train_loader:
        optimizer.zero_grad(set_to_none=True)
        graphs              = graphs.to(device, non_blocking=True)
        node_eigvals_target = node_eigvals_target.to(device, non_blocking=True)
        y = labels.to(device, non_blocking=True).float()
 
        with amp_ctx:
            out  = model(graphs, graphs.x)
            loss = loss_fn(out, node_eigvals_target) if encoding else loss_fn(out, y)
 
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
 
        total_train_loss += loss.item()
 
    # ---- validation (single pass, optionally compute AUC) ----
    model.eval()
    total_loss_val = 0.0
    all_preds  = [] if return_auc else None
    all_labels = [] if return_auc else None
 
    with torch.no_grad():
        for graphs, node_eigvals_target, labels in valid_loader:
            graphs              = graphs.to(device, non_blocking=True)
            node_eigvals_target = node_eigvals_target.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True).float()
 
            with amp_ctx:
                out  = model(graphs, graphs.x)
                loss = loss_fn(out, node_eigvals_target) if encoding else loss_fn(out, y)
 
            total_loss_val += loss.item()
            if return_auc:
                all_preds.append(out.view(-1).cpu())
                all_labels.append(y.view(-1).cpu())
 
    auc = None
    if return_auc:
        preds  = torch.cat(all_preds).numpy()
        lbls   = torch.cat(all_labels).numpy()
        auc    = roc_auc_score(lbls, preds)
 
    return total_train_loss, total_loss_val, auc
 
 
# ---------------------------------------------------------------------------
# Data preprocessing  (same as CPU version)
# ---------------------------------------------------------------------------
 
def compute_targets_with_graph(g, k=10, method="sum"):
    edge_index, edge_weight = get_laplacian(
        g.edge_index, normalization=None, num_nodes=g.num_nodes
    )
    L = to_dense_adj(edge_index, edge_attr=edge_weight,
                     max_num_nodes=g.num_nodes).squeeze(0)
    eigenvalues = torch.linalg.eigvalsh(L)[1:k + 1]
 
    if method == "sum":
        global_graph_feat = global_add_pool(g.x, g.batch)
    elif method == "avg":
        global_graph_feat = global_mean_pool(g.x, g.batch)
    else:
        raise ValueError("No valid feature selected")
    return g, eigenvalues, global_graph_feat
 
 
def pre_process_targets(graph_list, k=10, feat_method="sum", num_workers=None):
    """Parallel target computation (eigvalsh releases the GIL → thread-safe)."""
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, len(graph_list))
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(
            lambda g: compute_targets_with_graph(g, k=k, method=feat_method),
            graph_list
        ))
    return results
 
 
def pre_process_graphs(graph_list):
    """Aggregate edge features and precompute GCN normalisation.
 
    Stored on each graph as PreprocessedData so PyG batches edge_index_sl
    with the correct per-graph offsets.
    """
    processed = []
    for g in graph_list:
        g = g.clone()
 
        agg_edge_feat = scatter(
            g.edge_attr, g.edge_index[1],
            dim=0, dim_size=g.num_nodes, reduce='mean'
        )
        g.x = torch.cat([g.x, agg_edge_feat], dim=1)
 
        edge_index_sl, _ = add_self_loops(g.edge_index, num_nodes=g.num_nodes)
        row, col  = edge_index_sl
        deg       = degree(col, g.num_nodes, dtype=torch.float)
        d_inv_sq  = deg.pow(-0.5)
        d_inv_sq[d_inv_sq == float('inf')] = 0
        norm = d_inv_sq[row] * d_inv_sq[col]
 
        pd = PreprocessedData(
            x             = g.x,
            edge_index    = g.edge_index,
            edge_attr     = g.edge_attr,
            y             = g.y,
            num_nodes     = g.num_nodes,
            edge_index_sl = edge_index_sl,
            norm          = norm,
        )
        processed.append(pd)
    return processed
 
 
class GraphFeatureDataset(Dataset):
    def __init__(self, graph_list, eigval_list, feat_list, label_list):
        self.graphs = graph_list
        self.target = [
            torch.cat([eigval_list[i], feat_list[i].view(-1)], dim=0)
            for i in range(len(eigval_list))
        ]
        self.labels = label_list
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, index):
        return self.graphs[index], self.target[index], self.labels[index]
 
 
# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
 
def GAE_KAN_Script(batch_size, datafile, iterations, learning_rate, pred_epochs,
                   enc_epochs, num_harmonics, num_message_layers, num_readout_layers,
                   num_pred_layers, num_dec_layers, hidden_width, latent_size,
                   eval_every=5, patience=20, fast_cuda=True):
    """
    Parameters
    ----------
    eval_every : int
        Compute AUC every this many prediction epochs (default 5).
    patience : int
        Early-stop after this many epochs without AUC improvement (default 20).
    fast_cuda : bool
        If True, enables cudnn.benchmark (faster but non-deterministic).
        Set False if exact reproducibility is required.
    """
    print('GAE_KAN_GPU running...')
 
    torch.set_num_threads(os.cpu_count() or 1)
 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'  Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('  CUDA not available — falling back to CPU.')
 
    use_pin = (device.type == 'cuda')
    n_workers = min(4, max(0, (os.cpu_count() or 1) - 1))
    persist   = n_workers > 0
 
    # GradScaler for AMP (no-op if device is CPU)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
 
    recon_loss_fn = nn.L1Loss()
    pred_loss_fn  = nn.BCELoss()
 
    target_map = {'tox21': 12, 'muv': 17, 'sider': 27,
                  'clintox': 2, 'bace': 1, 'bbbp': 1, 'hiv': 1}
    file_name  = datafile.split("_")[0]
    target_dim = target_map[file_name]
 
    state = torch.load(datafile + '.pth', weights_only=False)
 
    # ---- preprocessing ----
    train_graphs = pre_process_graphs(state['train'])
    valid_graphs = pre_process_graphs(state['valid'])
 
    train_graph_targets = pre_process_targets(train_graphs)
    valid_graph_targets = pre_process_targets(valid_graphs)
 
    train_gs, train_evs, train_node_feat = zip(*train_graph_targets)
    valid_gs, valid_evs, valid_node_feat = zip(*valid_graph_targets)
 
    train_labels = [g.y for g in state['train']]
    valid_labels = [g.y for g in state['valid']]
 
    loader_bs = int(state['batch_size'])
 
    train_loader = DataLoader(
        GraphFeatureDataset(train_gs, train_evs, train_node_feat, train_labels),
        batch_size=loader_bs, shuffle=state['shuffle'], drop_last=True,
        num_workers=n_workers, persistent_workers=persist,
        pin_memory=use_pin,
    )
    valid_loader = DataLoader(
        GraphFeatureDataset(valid_gs, valid_evs, valid_node_feat, valid_labels),
        batch_size=loader_bs, shuffle=False, drop_last=True,
        num_workers=n_workers, persistent_workers=persist,
        pin_memory=use_pin,
    )
 
    node_dim = train_graphs[0].x.shape[1]
    k        = 10
    out_feat = k + node_dim
 
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # benchmark=True is faster but non-deterministic; respect fast_cuda flag
        torch.backends.cudnn.benchmark     = fast_cuda
        torch.backends.cudnn.deterministic = not fast_cuda
        np.random.seed(seed)
 
    All_AUC = []
    for i in range(iterations):
        best_auc         = 0.0
        epoch_since_best = 0
        set_seed(i)
 
        ae_model = KA_GAE(
            in_feat=node_dim, hidden_feat=hidden_width, latent_feat=latent_size,
            out_feat=out_feat, num_harmonics=num_harmonics,
            e_num_layers=num_message_layers, r_num_layers=num_readout_layers,
            d_num_layers=num_dec_layers, use_bias=True,
        ).to(device)
 
        latent_model = KA_latentpred(
            latent_feat=latent_size, hidden_feat=hidden_width, out_feat=target_dim,
            num_harmonics=num_harmonics, p_num_layers=num_pred_layers, use_bias=True,
        ).to(device)
 
        pred_model = LatentPass(ae_model.encoder, latent_model).to(device)
 
        ae_optimiser   = torch.optim.Adam(ae_model.parameters(),     lr=learning_rate)
        pred_optimiser = torch.optim.Adam(latent_model.parameters(), lr=learning_rate)
 
        # ---- encoder pre-training ----
        for _ in range(enc_epochs):
            train(ae_model, device, train_loader, valid_loader,
                  ae_optimiser, recon_loss_fn, encoding=True,
                  return_auc=False, scaler=scaler)
 
        # ---- predictor training with merged AUC ----
        AUC_list = []
        for epoch in range(pred_epochs):
            do_auc = (epoch % eval_every == 0) or (epoch == pred_epochs - 1)
            _, _, auc = train(pred_model, device, train_loader, valid_loader,
                              pred_optimiser, pred_loss_fn,
                              encoding=False, return_auc=do_auc, scaler=scaler)
            epoch_since_best += 1
            if auc is not None:
                AUC_list.append(auc)
                if auc > best_auc:
                    best_auc         = auc
                    epoch_since_best = 0
            if epoch_since_best > patience:
                break
 
        All_AUC.append(best_auc)
 
    return All_AUC