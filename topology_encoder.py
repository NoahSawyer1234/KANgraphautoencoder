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
    
class KA_autoencoder(nn.Module):
    def __init__(self, in_feat, hidden_feat, latent_feat, out_feat, num_harmonics, e_num_layers, use_bias=False):
        super(KA_autoencoder, self).__init__()
        encoder_modules = []
        encoder_modules.append(KAN_linear(in_feat, hidden_feat, num_harmonics, addbias=use_bias))
        for _ in range(e_num_layers - 1):
            encoder_modules.append(KAN_linear(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
        encoder_modules.append(KAN_linear(hidden_feat, latent_feat, num_harmonics, addbias=use_bias))
        self.encoder = nn.Sequential(*encoder_modules)

        self.decoder = nn.Sequential(
            KAN_linear(latent_feat, in_feat, num_harmonics, addbias=use_bias)
        )
        self.predictor = nn.Sequential( 
            KAN_linear(latent_feat,hidden_feat,num_harmonics),
            KAN_linear(hidden_feat, out_feat, num_harmonics),
            nn.Sigmoid()
        )

    def forward(self, features):
        z = self.encoder(features)
        out = self.decoder(z)
        pred = self.predictor(z)
        return out, pred
    
    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()
    
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


def train(model, device, train_loader, valid_loader, optimizer,recon_loss_fn,pred_loss_fn):
    model.train()
    total_train_loss = 0.0
    for labels, g in train_loader:
        y = labels.to(device)
        optimizer.zero_grad()     
        input_e = g.to(device)          
        out_e, latent_pred = model(input_e)                      
        recon_loss = recon_loss_fn(out_e, input_e)      
        pred_loss = pred_loss_fn(latent_pred.float(),y.float()) 
        loss = recon_loss + pred_loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()               

    model.eval()
    total_loss_val = 0.0
    with torch.no_grad():
        for labels, g in valid_loader:
            input_e = g.to(device)          
            out_e, latent_pred = model(input_e)                      
            recon_loss = recon_loss_fn(out_e, input_e)      
            pred_loss = pred_loss_fn(latent_pred.float(),y.float()) 
            loss = recon_loss + pred_loss
            total_loss_val += loss.item()
    return total_train_loss, total_loss_val


def predicting(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for labels, g in data_loader:
            y = labels.to(device)
            input_e = g.to(device)    
            out_e, pred = model(input_e)    
            all_preds.append(pred.view(-1))
            all_labels.append(y.view(-1))
    total_preds = torch.cat(all_preds, 0).cpu()
    total_labels = torch.cat(all_labels, 0).cpu()
    AUC = roc_auc_score(total_labels.numpy(), total_preds.numpy())
    return AUC

def collate_fn(batch):
    labels, eigvals = zip(*batch) 

    labels = torch.stack(labels)

    batched_graph = torch.stack(eigvals)

    return labels, batched_graph


if __name__ == '__main__':
    datafile = 'bace'

    recon_loss_fn = nn.L1Loss()
    pred_loss_fn = nn.BCELoss()

    batch_size = 128
    train_ratio = 0.8
    vali_ratio = 0.1
    test_ratio = 0.1
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    data_vec = ['bace','bbbp','hiv','clintox','sider','muv','tox21']
    target_dim = target_map[datafile]

    def pre_process_lap_eigenvectors(graph_list):
        processed_graphs = []
        for g in graph_list:
            eigenvectors, eigenvalues = dgl.lap_pe(g, k=10, return_eigval=True)
            processed_graphs.append(eigenvalues)
        return processed_graphs

    state = torch.load(datafile+'128_811.pth')

    train_eigval_vec = pre_process_lap_eigenvectors(state['train_graph_list'])
    valid_eigval_vec  = pre_process_lap_eigenvectors(state['valid_graph_list'])
    test_eigval_vec  = pre_process_lap_eigenvectors(state['test_graph_list'])

    loaded_train_dataset = CustomDataset(state['train_label'], train_eigval_vec)
    loaded_valid_dataset = CustomDataset(state['valid_label'], valid_eigval_vec)
    loaded_test_dataset = CustomDataset(state['test_label'], test_eigval_vec) 

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

    iters = 10
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

        model = KA_autoencoder(in_feat=10, hidden_feat=64, latent_feat=128, out_feat=1, 
                        num_harmonics=num_harmonics, e_num_layers=num_layers, use_bias=False)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        #scheduler = StepLR(optimiser, step_size=100, gamma=0.1)
        for epoch in range(epochs):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimiser,
                                         recon_loss_fn=recon_loss_fn, pred_loss_fn=pred_loss_fn)
            AUC = predicting(model, device, loaded_test_loader)
            loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)
            AUC_list.append(AUC)
            if epoch % 10 == 0 :
                print('Epoch - ', epoch)
                print('Train loss - ', train_loss)
                print('Valid loss - ', vali_loss)
                print('AUC - ', AUC)
        #torch.save(model.state_dict(), 'model.pth')
        if i == iters -1: 
            plt.plot(AUC_list)
            plt.savefig('AUCplot_topo.png')
            plt.show()
            plt.plot(loss_list)
            plt.savefig('Lossplot_topo.png')
            plt.show()
            plt.plot(vali_loss_list)
            plt.savefig('ValidLossplot_topo.png')
            plt.show()
        All_AUC.append(max(AUC_list))
        print('Iteration done!')
    print(All_AUC)