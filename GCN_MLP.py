
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

class MLP_message_passing(MessagePassing):
    def __init__(self, input_size, output_size, addbias=True):
        super().__init__(aggr='add')
        self.lin = nn.Linear(input_size,output_size, bias=False)
        self.addbias = addbias
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self,x,edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
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

class MLP_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_message_layers, use_bias=False):
        super(MLP_GCN, self).__init__()
        self.num_message_layers = num_message_layers
        #self.batch_norms = nn.ModuleList()
        #self.dropout = nn.Dropout(0.2)
        self.node_embedding = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.message_layers = nn.ModuleList()

        for _ in range(num_message_layers):
            self.message_layers.append(MLP_message_passing(hidden_size, hidden_size, addbias=use_bias))
            #self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
        self.lr = nn.LeakyReLU()
        
        self.meanpool = global_mean_pool
        self.readout_layers =  nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=use_bias),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size, bias=use_bias),
            nn.Sigmoid()
            )
    def forward(self, g, features):
        h = self.node_embedding(features)
        h = self.lr(h)
        #h = self.dropout(h)
        for layer in self.message_layers:
            h = layer(h,g.edge_index)  
            h = self.lr(h)
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

def GCN_MLP_Script(batch_size, datafile, iterations, learning_rate, num_epochs, 
                   num_message_layers, hidden_width):
    datafile = datafile
    loss_fn = nn.BCELoss(reduction='mean')

    batch_size = batch_size
    iters = iterations
    lr = learning_rate
    epochs = num_epochs
    num_message_layers = num_message_layers

    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    target_dim = target_map[datafile]

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
    print('dataset was loaded!')
    node_dim = train_graphs[0].x.shape[1]
    print(f"node feature dim after pre-processing: {node_dim}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU!!!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    All_AUC = []
    for i in range(iters):
        set_seed(i)
        print('Iteration - ', i+1)
        AUC_list = []
        model = MLP_GCN(input_size= 23+10 , hidden_size=hidden_width, output_size=target_dim
                        , num_message_layers=num_message_layers, use_bias=True)
        #total_params = sum(p.numel() for p in model.parameters())
        #print(f"Total parameters: {total_params}")
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            train_loss,vali_loss = train(model, device, train_loader, valid_loader, optimiser,loss_fn=loss_fn)
            AUC = predicting(model, device, valid_loader)
            AUC_list.append(AUC)
        All_AUC.append(max(AUC_list))
    return All_AUC


'''        if i == iters-1:
            plt.plot(AUC_list)
            plt.savefig('AUCplot_GCN_MLP.png')
            plt.show()
            plt.plot(loss_list)
            plt.savefig('Lossplot_GCN_MLP.png')
            plt.show()
            plt.plot(vali_loss_list)
            plt.savefig('ValidLossplot_GCN_MLP.png')
            plt.show()
            All_AUC.append(max(AUC_list))
            print('Iteration done!')
        '''