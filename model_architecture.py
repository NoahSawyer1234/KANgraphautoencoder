import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import statistics
from torch.utils.data import DataLoader
from dgl.nn import SumPooling, AvgPooling, MaxPooling
from sklearn.metrics import roc_auc_score
import time
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class KAN_linear(nn.Module):
    def __init__(self, indim, outdim, num_harmonics, addbias=True):
        super(KAN_linear,self).__init__()
        self.gridsize= num_harmonics
        self.addbias = addbias
        self.inputdim = indim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, indim, num_harmonics) / 
                                             (np.sqrt(indim) * np.sqrt(self.gridsize)))
        k = torch.arange(1, num_harmonics + 1).view(1, 1, 1, num_harmonics)
        self.register_buffer('k', k)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
            orig_shape = x.shape
            x = x.view(-1, self.inputdim) 
            x_scaled = x.unsqueeze(1).unsqueeze(-1) * self.k
            cos_x = torch.cos(x_scaled)
            sin_x = torch.sin(x_scaled)
            y_cos = torch.einsum('njih,oih->no', cos_x, self.fouriercoeffs[0])
            y_sin = torch.einsum('njih,oih->no', sin_x, self.fouriercoeffs[1])            
            y = y_cos + y_sin            
            if self.addbias:
                y += self.bias
            return y.view(*orig_shape[:-1], self.outdim)


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_harmonics, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = num_harmonics
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.fouriercoeffs = nn.Parameter(torch.randn(2, out_feats, in_feats, num_harmonics) / 
                                          (np.sqrt(in_feats) * np.sqrt(num_harmonics)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))

        k = torch.arange(1, num_harmonics + 1).view(1, 1, 1, num_harmonics)
        self.register_buffer('k', k)


    def forward(self, g, x):
        x_scaled = x.unsqueeze(-1).unsqueeze(-1) * self.k
        
        cos_x = torch.cos(x_scaled)
        sin_x = torch.sin(x_scaled)
        
        h_cos = torch.einsum('njih,oih->no', cos_x, self.fouriercoeffs[0])
        h_sin = torch.einsum('njih,oih->no', sin_x, self.fouriercoeffs[1])
        h_transformed = h_cos + h_sin

        with g.local_scope():
            g.ndata['h_proc'] = h_transformed
            g.update_all(fn.copy_u('h_proc', 'm'), fn.sum('m', 'h_final'))
            res = g.ndata['h_final']
            if self.addbias:
                res += self.bias
            return res
    

class KA_GNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, num_harmonics, num_layers, pooling, use_bias=False):
        super(KA_GNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.kan_line = KAN_linear(in_feat, hidden_feat, num_harmonics, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
            self.batch_norms.append(nn.BatchNorm1d(hidden_feat))
       
        self.linear_1 = KAN_linear(hidden_feat, out_feat, num_harmonics, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, num_harmonics, addbias=use_bias)
        self.linear = KAN_linear(hidden_feat, out, num_harmonics, addbias=use_bias)

        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        layers_kan = [self.linear_1,
                        self.leaky_relu,
                        self.dropout,
                        self.linear_2,
                        nn.Sigmoid()]
        
        self.Readout = nn.Sequential(*layers_kan)  

    def forward(self, g, features):
        h = self.kan_line(features)
        h = self.dropout(h)
        for layer, batch_norm in zip(self.layers,self.batch_norms):
            h = layer(g, h)  
            h = batch_norm(h) 
            h = self.dropout(h)  
        y = self.avgpool(g, h)

        out = self.Readout(y)     
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()


def message_func(edges):
    return {'feat': edges.data['feat']}

def reduce_func(nodes):
    num_edges = nodes.mailbox['feat'].size(1)  
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
    return {'agg_feats': agg_feats}

def update_node_features(g):
    g.send_and_recv(g.edges(), message_func, reduce_func)
    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)
    return g

def train(model, device, train_loader, valid_loader, optimizer,loss_fn):
    model.train()
    total_train_loss = 0.0
    for labels, g in train_loader:
        optimizer.zero_grad()
        y = labels.to(device)           
        g = g.to(device) 
        x = g.ndata['feat']                
        out = model(g, x)                      
        loss = loss_fn(out, y.float())            
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()               

    model.eval()
    total_loss_val = 0.0
    with torch.no_grad():
        for labels, g in valid_loader:
            y = labels.to(device)
            g = g.to(device)
            x = g.ndata['feat']
            out = model(g, x)
            loss = loss_fn(out, y.float())      
            total_loss_val += loss.item()
    return total_train_loss, total_loss_val

def predicting(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for labels, g in data_loader:
            y = labels.to(device)
            g = g.to(device)
            x = g.ndata['feat']
            output = model(g, x)
            all_preds.append(output.view(-1))
            all_labels.append(y.view(-1))
    total_preds = torch.cat(all_preds, 0).cpu()
    total_labels = torch.cat(all_labels, 0).cpu()
    AUC = roc_auc_score(total_labels.numpy(), total_preds.numpy())
    return AUC

def collate_fn(batch):
    labels, graphs = zip(*batch) 

    labels = torch.stack(labels)

    batched_graph = dgl.batch(graphs)

    return labels, batched_graph

class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        

        graph = self.graphs[index].to(self.device)
        
        return label, graph

#### Do node embedding before rather than every iteration
def pre_process_graphs(graph_list):
    processed_graphs = []
    for g in graph_list:
        g.update_all(fn.copy_e('feat', 'm'), fn.mean('m', 'agg_edge_feat'))
        combined_feat = torch.cat([g.ndata['feat'], g.ndata['agg_edge_feat']], dim=1)
        g.ndata['feat'] = combined_feat
        del g.ndata['agg_edge_feat']
        processed_graphs.append(g)
    return processed_graphs

#################   Load dataset in, change loss for different datasets
if __name__ == '__main__':
    datafile = 'bace'
    loss_fn = nn.BCELoss(reduction='mean')

    batch_size = 128
    train_ratio = 0.8
    vali_ratio = 0.1
    test_ratio = 0.1
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    data_vec = ['bace','bbbp','hiv','clintox','sider','muv','tox21']
    target_dim = target_map[datafile]

    state = torch.load(datafile+'128_811.pth')

    train_graphs = pre_process_graphs(state['train_graph_list'])
    valid_graphs = pre_process_graphs(state['valid_graph_list'])
    test_graphs = pre_process_graphs(state['test_graph_list'])

    loaded_train_dataset = CustomDataset(state['train_label'], train_graphs)
    loaded_valid_dataset = CustomDataset(state['valid_label'], valid_graphs)
    loaded_test_dataset = CustomDataset(state['test_label'], test_graphs)

    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=int(state['batch_size']), shuffle=state['shuffle'],num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=int(state['batch_size']), shuffle=state['shuffle'],num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=int(state['batch_size']), shuffle=state['shuffle'],num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    print('dataset was loaded!')

    print("length of training set:",len(loaded_train_dataset))
    print("length of validation set:",len(loaded_valid_dataset))
    print("length of testing set:",len(loaded_test_dataset))

    iters = 1
    lr = 10**-4
    epochs = 500
    num_harmonics = 1
    num_layers = 4
    pooling = "avg"
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    All_AUC = []

    start_time = time.time()

    for i in range(iters):
        set_seed(i)
        print('Iteration - ', i+1)
        AUC_list = []
        loss_list = []
        vali_loss_list = []

        model = KA_GNN(in_feat=92+18, hidden_feat=64, out_feat=32, out=target_dim, 
                        num_harmonics=num_harmonics, num_layers=num_layers, pooling = pooling, use_bias=False)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = StepLR(optimiser, step_size=100, gamma=0.1)
        for epoch in range(epochs):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimiser,loss_fn=loss_fn)
            #scheduler.step()
            #print('AUC - ', AUC, ' ---------- LR - ', optimiser.param_groups[0]['lr'])
            AUC = predicting(model, device, loaded_test_loader)
            loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)
            AUC_list.append(AUC)
            if epoch % 10 == 0 :
                print('Epoch - ', epoch, ' AUC - ', AUC)
                print('Train loss - ', train_loss)
                print('Valid loss - ', vali_loss)
                print('AUC - ', AUC)
        torch.save(model.state_dict(), 'model.pth')
        plt.plot(AUC_list)
        plt.savefig('AUCplot.png')
        plt.show()
        plt.plot(loss_list)
        plt.savefig('Lossplot.png')
        plt.show()
        plt.plot(vali_loss_list)
        plt.savefig('ValidLossplot.png')
        plt.show()
        All_AUC.append(max(AUC_list))
        print('Iteration done!')
        
    print(All_AUC)
    mean_value = statistics.mean(All_AUC)

    #std_dev = statistics.stdev(All_AUC)

    print("mean:", mean_value)
    #print("std:", std_dev)

#GitLens