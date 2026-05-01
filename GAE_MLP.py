# This architecture uses a GCN encoder, with a decoder that produces the k=10 smallest non-trivial eigenvalues
# of the graph laplacian (topological info) concatenated with the sum of all node vectors (post-node embedding, 
# ie. includes edge info) 

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import time
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import scatter
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch.utils.data import Dataset

class KAN_node_embedding(nn.Module):
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super(KAN_node_embedding,self).__init__()
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

# Adapted from https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/create_gnn.html
class KAN_message_passing(MessagePassing):
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super().__init__(aggr='add')
        self.KAN = KAN_node_embedding(input_size,output_size,num_harmonics, addbias=False)
        self.addbias = addbias
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self,x,edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.KAN(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        y = self.propagate(edge_index, x=x, norm=norm)
        if self.addbias:
            y = y + self.bias
        return y
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class KA_GCN_latent(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_harmonics, num_message_layers, use_bias=False):
        super(KA_GCN_latent, self).__init__()
        self.num_message_layers = num_message_layers
        #self.batch_norms = nn.ModuleList()
        #self.dropout = nn.Dropout(0.2)

        self.node_embedding = KAN_node_embedding(input_size, hidden_size, num_harmonics, addbias=use_bias)
        self.message_layers = nn.ModuleList()
        for _ in range(num_message_layers):
            self.message_layers.append(KAN_message_passing(hidden_size, hidden_size, num_harmonics, addbias=use_bias))
            #self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
        
        self.meanpool = global_mean_pool
        self.latent_readout = KAN_node_embedding(hidden_size, latent_size, num_harmonics, addbias=use_bias)

    def forward(self, g, features):
        h = self.node_embedding(features)
        #h = self.dropout(h)
        for layer in self.message_layers:
            h = layer(h,g.edge_index)  
            #h = batch_norm(h) 
            #h = self.dropout(h)  
        y = global_mean_pool(h,g.batch)
        out = self.latent_readout(y)     
        return out

class KA_GAE(nn.Module):
    def __init__(self, in_feat, hidden_feat, latent_feat, out_feat, num_harmonics, e_num_layers, use_bias=False):
        super(KA_GAE, self).__init__() 
        self.encoder = KA_GCN_latent(in_feat,hidden_feat,latent_feat, num_harmonics, e_num_layers, use_bias=use_bias)
        self.decoder = KAN_node_embedding(latent_feat, out_feat, num_harmonics, addbias=use_bias)
    def forward(self, g, features):
        z = self.encoder(g, features)
        out = self.decoder(z)
        return out

class KA_latentpred(nn.Module):
    def __init__(self, latent_feat, hidden_feat, out_feat, num_harmonics, p_num_layers, use_bias= True):
        super(KA_latentpred, self).__init__()
        # Need to update this if I want variable latent predictor
        # Need to figure out bias too
        pred_modules = [KAN_node_embedding(latent_feat, hidden_feat, num_harmonics, use_bias)]
        for _ in range(p_num_layers - 2):
            pred_modules.append(KAN_node_embedding(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
        pred_modules.append(KAN_node_embedding(hidden_feat, out_feat, num_harmonics,use_bias))
        pred_modules.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*pred_modules)
    def forward(self, latent):
        return self.predictor(latent)

class LatentPass(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, g, x):
        z = self.encoder(g,x)
        return self.predictor(z)

def train(model, device, train_loader, valid_loader, optimizer, loss_fn, encoding=True):
    model.train()
    total_train_loss = 0.0
    for graphs, node_eigvals_target, labels in train_loader:
        optimizer.zero_grad()
        graphs = graphs.to(device)
        node_eigvals_target = node_eigvals_target.to(device)
        y = labels.to(device).float()
        out = model(graphs, graphs.x)
        loss = loss_fn(out, node_eigvals_target) if encoding else loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_loss_val = 0.0
    with torch.no_grad():
        for graphs, node_eigvals_target, labels in valid_loader:
            graphs = graphs.to(device)
            node_eigvals_target = node_eigvals_target.to(device)
            y = labels.to(device).float()
            out = model(graphs, graphs.x)
            loss = loss_fn(out, node_eigvals_target) if encoding else loss_fn(out, y)
            total_loss_val += loss.item()

    return total_train_loss, total_loss_val


def predicting(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for graphs, eigvals, labels in data_loader:
            graphs = graphs.to(device)
            eigvals = eigvals.to(device)
            y = labels.to(device).float()
            out = model(graphs, graphs.x)
            all_preds.append(out.view(-1))
            all_labels.append(y.view(-1))

    total_preds = torch.cat(all_preds, 0).cpu()
    total_labels = torch.cat(all_labels, 0).cpu()
    AUC = roc_auc_score(total_labels.numpy(), total_preds.numpy())
    return AUC

def compute_targets_with_graph(g, k=10, method = "sum"):
    edge_index, edge_weight = get_laplacian(
        g.edge_index,
        normalization=None,   
        num_nodes=g.num_nodes
    )
    L = to_dense_adj(edge_index, edge_attr=edge_weight,
                     max_num_nodes=g.num_nodes).squeeze(0)
    eigenvalues = torch.linalg.eigvalsh(L)
    eigenvalues = eigenvalues[1:k+1]
    if method == "sum":
        global_graph_feat = global_add_pool(g.x, g.batch)
    elif method == "avg":
        global_graph_feat = global_mean_pool(g.x, g.batch)
    else:
        ValueError("No valid fetaure selected")
    return g, eigenvalues, global_graph_feat

def pre_process_targets(graph_list, k=10, feat_method="sum"):
    return [compute_targets_with_graph(g, k=k, method=feat_method) for g in graph_list]


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
    
#### Do node embedding before rather than every iteration
def pre_process_graphs(graph_list):
    processed_graphs = []
    for g in graph_list:
        g = g.clone()
        agg_edge_feat = scatter(
            g.edge_attr,           
            g.edge_index[1],     
            dim=0,
            dim_size=g.num_nodes, 
            reduce='mean'         
        )                        
        g.x = torch.cat([g.x, agg_edge_feat], dim=1)  
        processed_graphs.append(g)
    return processed_graphs

def GAE_KAN_Script(batch_size, datafile, iterations, learning_rate, pred_epochs,enc_epochs, 
                   num_message_layers, num_pred_layers, hidden_width,latent_size):
    print('GAE_KAN running...')
    datafile = datafile
    recon_loss_fn = nn.L1Loss()
    pred_loss_fn = nn.BCELoss()
    batch_size = batch_size
    iters = iterations
    lr = learning_rate
    epochs = pred_epochs
    encoding_epochs = enc_epochs
    num_harmonics = num_harmonics
    num_enc_layers = num_message_layers
    num_pred_layers = num_pred_layers

    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    file_name = datafile.split("_")[0]
    target_dim = target_map[file_name]

    state = torch.load(datafile+'.pth')

    #Do node embedding first
    train_graphs = pre_process_graphs(state['train'])
    valid_graphs = pre_process_graphs(state['valid'])
    test_graphs = pre_process_graphs(state['test'])

    #Get (graphs, eigenvalue) tuple
    train_graph_targets = pre_process_targets(train_graphs)
    valid_graph_targets = pre_process_targets(valid_graphs)
    test_graph_targets  = pre_process_targets(test_graphs)

    # Need to unzip for batches
    train_gs, train_evs, train_node_feat = zip(*train_graph_targets)
    valid_gs, valid_evs, valid_node_feat = zip(*valid_graph_targets)
    #test_gs, test_evs, test_node_feat = zip(*test_graph_targets)
    
    train_labels = [g.y for g in state['train']]
    valid_labels = [g.y for g in state['valid']]
    #test_labels  = [g.y for g in state['test']]

    train_loader = DataLoader(
        GraphFeatureDataset(train_gs, train_evs, train_node_feat, train_labels), 
        batch_size=int(state['batch_size']), 
        shuffle=state['shuffle'], 
        drop_last=True
    )

    valid_loader = DataLoader(
        GraphFeatureDataset(valid_gs, valid_evs, valid_node_feat, valid_labels), 
        batch_size=int(state['batch_size']), 
        shuffle=False,  # Shuffling validation data is generally not recommended
        drop_last=True
    )

    '''    test_loader = DataLoader(
        GraphFeatureDataset(test_gs, test_evs, test_node_feat, test_labels), 
        batch_size=int(state['batch_size']), 
        shuffle=False,  # Keep test order consistent
        drop_last=True
    )'''

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    #print('dataset was loaded!')
    node_dim = train_graphs[0].x.shape[1]
    #print(f"node feature dim after pre-processing: {node_dim}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        #print('The code uses GPU!!!')
    else:
        device = torch.device('cpu')
        #print('The code uses CPU!!!')

    All_AUC = []
    for i in range(iters):
        print(i)
        set_seed(i)
        #print('Iteration -', i + 1)
        ae_model = KA_GAE(in_feat=23+10, hidden_feat=hidden_width, latent_feat=latent_size, out_feat=10 + 23 + 10,
                                  num_harmonics=num_harmonics, e_num_layers=num_enc_layers,
                                  use_bias=True).to(device)
        latent_model = KA_latentpred(latent_feat=latent_size, hidden_feat=hidden_width, out_feat=target_dim,
                                     num_harmonics=num_harmonics,p_num_layers=num_pred_layers, use_bias=True).to(device)
        
        pred_model = LatentPass(ae_model.encoder, latent_model).to(device)

        #total_params = (sum(p.numel() for p in ae_model.parameters()) +
                        #sum(p.numel() for p in latent_model.parameters()))
        #print(f"Total parameters: {total_params}")

        ae_optimiser   = torch.optim.Adam(ae_model.parameters(), lr=lr)
        pred_optimiser = torch.optim.Adam(latent_model.parameters(), lr=lr)

        for epoch in range(encoding_epochs):
            train_loss, vali_loss = train(ae_model, device, train_loader, valid_loader,
                                          ae_optimiser, recon_loss_fn, encoding=True)
        AUC_list = []
        for epoch in range(epochs):
            train_loss, vali_loss = train(pred_model, device, train_loader, valid_loader,
                                          pred_optimiser, pred_loss_fn, encoding=False)
            AUC = predicting(pred_model, device, valid_loader)
            AUC_list.append(AUC)
        All_AUC.append(max(AUC_list))
    return All_AUC


'''        if i == iters - 1:
            for data, name in [(AUC_list, 'AUC'), (loss_list, 'Loss'), (vali_loss_list, 'ValidLoss')]:
                plt.plot(data)
                plt.title(f'{name} vs Epoch')
                plt.savefig(f'{name}plot_GAE_KAN.png')
                plt.show()'''
