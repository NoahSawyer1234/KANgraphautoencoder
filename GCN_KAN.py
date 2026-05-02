
import torch
import torch.nn as nn
import numpy as np
import statistics
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import time
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import scatter
import json

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
        cos_terms = torch.cos(x_scaled)  # [batch, 1, in_dim, num_harmonics]
        sin_terms = torch.sin(x_scaled)  # [batch, 1, in_dim, num_harmonics]
        y_cos = torch.einsum('bnih,oih->bo', cos_terms, self.fouriercoeffs[0])
        y_sin = torch.einsum('bnih,oih->bo', sin_terms, self.fouriercoeffs[1])
        y = y_cos + y_sin  # [batch, out_dim]
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

class KA_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_harmonics, num_message_layers, num_readout_layers, use_bias=False):
        super(KA_GCN, self).__init__()
        self.num_message_layers = num_message_layers
        #self.batch_norms = nn.ModuleList()
        #self.dropout = nn.Dropout(0.2)

        self.node_embedding = KAN_node_embedding(input_size, hidden_size, num_harmonics, addbias=use_bias)
        self.message_layers = nn.ModuleList()
        for _ in range(num_message_layers):
            self.message_layers.append(KAN_message_passing(hidden_size, hidden_size, num_harmonics, addbias=use_bias))
            #self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
        
        self.meanpool = global_mean_pool
        self.readout_layers = nn.ModuleList()
        if num_readout_layers ==1:
            self.readout_layers.append(KAN_node_embedding(hidden_size,output_size,num_harmonics,addbias=use_bias))
        else:
            for _ in range(num_readout_layers-1):
                self.readout_layers.append(KAN_node_embedding(hidden_size, hidden_size, num_harmonics, addbias=use_bias))
                #self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
            self.readout_layers.append(KAN_node_embedding(hidden_size, output_size ,num_harmonics,addbias=use_bias))
        self.readout_layers.append(nn.Sigmoid())
        self.readout_layers = nn.Sequential(*self.readout_layers)

    def forward(self, g, features):
        h = self.node_embedding(features)
        #h = self.dropout(h)
        for layer in self.message_layers:
            h = layer(h,g.edge_index)  
            #h = batch_norm(h) 
            #h = self.dropout(h)  
        y = global_mean_pool(h,g.batch)
        out = self.readout_layers(y)     
        return out

def train(model, device, train_loader, valid_loader, optimizer, loss_fn):
    model.train()
    total_train_loss = 0.0
    for g in train_loader:                  
        optimizer.zero_grad()
        g = g.to(device)
        y = g.y                               
        x = g.x                              
        out = model(g, x).squeeze(-1)
        loss = loss_fn(out, y.float())
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_loss_val = 0.0
    with torch.no_grad():
        for g in valid_loader:
            g = g.to(device)
            y = g.y
            x = g.x
            out = model(g, x).squeeze(-1)
            loss = loss_fn(out, y.float())
            total_loss_val += loss.item()

    return total_train_loss, total_loss_val

def predicting(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for g in data_loader:
            g = g.to(device)
            y = g.y
            x = g.x
            output = model(g, x).squeeze(-1)
            all_preds.append(output.view(-1))
            all_labels.append(y.view(-1))

    total_preds = torch.cat(all_preds, 0).cpu()
    total_labels = torch.cat(all_labels, 0).cpu()
    AUC = roc_auc_score(total_labels.numpy(), total_preds.numpy())
    return AUC

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

def GCN_KAN_Script(batch_size, datafile, iterations, learning_rate, num_epochs,num_harmonics, 
                   num_message_layers, num_readout_layers, hidden_width):
    datafile = datafile
    loss_fn = nn.BCELoss(reduction='mean')

    batch_size = batch_size
    iters = iterations
    lr = learning_rate
    epochs = num_epochs
    num_harmonics = num_harmonics
    num_message_layers = num_message_layers
    num_readout_layers =  num_readout_layers

    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    file_name = datafile.split("_")[0]
    target_dim = target_map[file_name]

    state = torch.load(datafile+'.pth')

    train_graphs = pre_process_graphs(state['train'])
    valid_graphs = pre_process_graphs(state['valid'])
    test_graphs = pre_process_graphs(state['test'])
    
    train_loader = DataLoader(train_graphs, batch_size=int(state['batch_size']), shuffle=state['shuffle'],  num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_graphs, batch_size=int(state['batch_size']), shuffle=False, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_graphs,  batch_size=int(state['batch_size']), shuffle=False, num_workers=0, drop_last=True)

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
        set_seed(i)
        #print('Iteration - ', i+1)
        AUC_list = []
        model = KA_GCN(input_size= 23+10 , hidden_size=hidden_width, output_size=target_dim,
                        num_harmonics=num_harmonics, num_message_layers=num_message_layers,
                        num_readout_layers= num_readout_layers, use_bias=True)
        #total_params = sum(p.numel() for p in model.parameters())
        #print(f"Total parameters: {total_params}")
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            train_loss,vali_loss = train(model, device, train_loader, valid_loader, optimiser,loss_fn=loss_fn)
            AUC = predicting(model, device, valid_loader)
            AUC_list.append(AUC)
        All_AUC.append(max(AUC))
    return All_AUC


'''        if i == iters-1:
            plt.plot(AUC_list)
            plt.savefig('AUCplot_GCN_KAN.png')
            plt.show()
            plt.plot(loss_list)
            plt.savefig('Lossplot_GCN_KAN.png')
            plt.show()
            plt.plot(vali_loss_list)
            plt.savefig('ValidLossplot_GCN_KAN.png')
            plt.show()
            All_AUC.append(max(AUC_list))
            print('Iteration done!')'''