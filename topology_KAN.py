import torch
import torch.nn as nn
import numpy as np
import statistics
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch_geometric.utils import get_laplacian, to_dense_adj


class KAN_layer(nn.Module):
    def __init__(self, input_size, output_size, num_harmonics, addbias=True):
        super(KAN_layer,self).__init__()
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

class KA_autoencoder(nn.Module):
    def __init__(self, in_feat, hidden_feat, latent_feat, num_harmonics, e_num_layers, use_bias=False):
        super(KA_autoencoder, self).__init__() 

        encoder_modules = []
        encoder_modules.append(KAN_layer(in_feat, hidden_feat, num_harmonics, addbias=use_bias))
        for _ in range(e_num_layers - 2):
            encoder_modules.append(KAN_layer(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
        encoder_modules.append(KAN_layer(hidden_feat, latent_feat, num_harmonics, addbias=use_bias))
        self.encoder = nn.Sequential(*encoder_modules)

        #Need to edit this if I want deeper decoder
        self.decoder = nn.Sequential(
            KAN_layer(latent_feat, in_feat, num_harmonics, addbias=use_bias)
        )

    def forward(self, features):
        z = self.encoder(features)
        out = self.decoder(z)
        return out

class KA_latentpred(nn.Module):
    def __init__(self, latent_feat, hidden_feat, out_feat, num_harmonics, p_num_layers, use_bias= True):
        super(KA_latentpred, self).__init__()
        # Need to update this if I want variable latent predictor
        # Need to figure out bias too
        pred_modules = [KAN_layer(latent_feat, hidden_feat, num_harmonics, use_bias)]
        for _ in range(p_num_layers - 2):
            pred_modules.append(KAN_layer(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
        pred_modules.append(KAN_layer(hidden_feat, out_feat, num_harmonics,use_bias))
        pred_modules.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*pred_modules)
    def forward(self, latent):
        return self.predictor(latent)

# For prediction, gets fed AE and latentpred, and freezes encoder 
class LatentPass(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, x):
        z = self.encoder(x)
        return self.predictor(z)

class EigenvalueDataset(Dataset):
    def __init__(self, eigval_list, label_list):
        self.eigvals = eigval_list
        self.labels = label_list
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.eigvals[index], self.labels[index]


def compute_lap_eigenvalues(g, k=10):
    edge_index, edge_weight = get_laplacian(
        g.edge_index,
        normalization=None,   
        num_nodes=g.num_nodes
    )
    L = to_dense_adj(edge_index, edge_attr=edge_weight,
                     max_num_nodes=g.num_nodes).squeeze(0)
    eigenvalues = torch.linalg.eigvalsh(L)
    eigenvalues = eigenvalues[1:k+1]
    return eigenvalues

def pre_process_lap_eigenvectors(graph_list, k=10):
    return [compute_lap_eigenvalues(g, k=k) for g in graph_list]


def train(model, device, train_loader, valid_loader, optimizer, loss_fn, encoding=True):
    model.train()
    total_train_loss = 0.0
    for eigvals, labels in train_loader:
        optimizer.zero_grad()
        eigvals = eigvals.to(device)
        y = labels.to(device).float()
        out = model(eigvals)
        loss = loss_fn(out, eigvals) if encoding else loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_loss_val = 0.0
    with torch.no_grad():
        for eigvals, labels in valid_loader:
            eigvals = eigvals.to(device)
            y = labels.to(device).float()
            out = model(eigvals)
            loss = loss_fn(out, eigvals) if encoding else loss_fn(out, y)
            total_loss_val += loss.item()

    return total_train_loss, total_loss_val


def predicting(model, device, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for eigvals, labels in data_loader:
            eigvals = eigvals.to(device)
            y = labels.to(device)
            out = model(eigvals)
            all_preds.append(out.view(-1))
            all_labels.append(y.view(-1))
    total_preds = torch.cat(all_preds, 0).cpu()
    total_labels = torch.cat(all_labels, 0).cpu()
    return roc_auc_score(total_labels.numpy(), total_preds.numpy())


if __name__ == '__main__':
    datafile = 'bace'
    recon_loss_fn = nn.L1Loss()
    pred_loss_fn = nn.BCELoss()
    batch_size = 128
    vali_ratio = 0.1
    target_map = {'tox21': 12, 'muv': 17, 'sider': 27, 'clintox': 2,
                  'bace': 1, 'bbbp': 1, 'hiv': 1}
    target_dim = target_map[datafile]

    # load from new PyG format — labels are attached to graph objects as .y
    state = torch.load(datafile + '.pth')

    train_eigvals = pre_process_lap_eigenvectors(state['train'])
    valid_eigvals = pre_process_lap_eigenvectors(state['valid'])
    test_eigvals  = pre_process_lap_eigenvectors(state['test'])

    # extract labels from graph objects
    train_labels = [g.y for g in state['train']]
    valid_labels = [g.y for g in state['valid']]
    test_labels  = [g.y for g in state['test']]

    # plain PyTorch DataLoader is fine here — inputs are tensors not graphs
    train_loader = DataLoader(EigenvalueDataset(train_eigvals, train_labels),
                              batch_size=int(state['batch_size']),
                              shuffle=state['shuffle'], drop_last=True)
    valid_loader = DataLoader(EigenvalueDataset(valid_eigvals, valid_labels),
                              batch_size=int(state['batch_size']),
                              shuffle=False, drop_last=True)
    test_loader  = DataLoader(EigenvalueDataset(test_eigvals, test_labels),
                              batch_size=int(state['batch_size']),
                              shuffle=False, drop_last=True)

    print('dataset was loaded!')

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using:', device)

    iters = 1
    lr = 1e-4
    epochs = 1000
    encoding_epochs = int(epochs/2)
    num_harmonics = 1
    num_enc_layers = 5
    num_pred_layers = 5
    All_AUC = []

    for i in range(iters):
        set_seed(i)
        print('Iteration -', i + 1)

        ae_model = KA_autoencoder(in_feat=10, hidden_feat=64, latent_feat=128,
                                  num_harmonics=num_harmonics, e_num_layers=num_enc_layers,
                                  use_bias=True).to(device)
        latent_model = KA_latentpred(latent_feat=128, hidden_feat=64, out_feat=1,
                                     num_harmonics=num_harmonics,p_num_layers=num_pred_layers, use_bias=True).to(device)
        pred_model = LatentPass(ae_model.encoder, latent_model).to(device)

        total_params = (sum(p.numel() for p in ae_model.parameters()) +
                        sum(p.numel() for p in latent_model.parameters()))
        print(f"Total parameters: {total_params}")

        ae_optimiser   = torch.optim.Adam(ae_model.parameters(), lr=lr)
        pred_optimiser = torch.optim.Adam(latent_model.parameters(), lr=lr)

        for epoch in range(encoding_epochs):
            train_loss, vali_loss = train(ae_model, device, train_loader, valid_loader,
                                          ae_optimiser, recon_loss_fn, encoding=True)
            if epoch % 10 == 0:
                print(f'AE Epoch {epoch} — train: {train_loss:.4f}  valid: {vali_loss:.4f}')

        loss_list = []
        vali_loss_list = []
        AUC_list = []
        for epoch in range(epochs):
            train_loss, vali_loss = train(pred_model, device, train_loader, valid_loader,
                                          pred_optimiser, pred_loss_fn, encoding=False)
            AUC = predicting(pred_model, device, valid_loader)
            loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)
            AUC_list.append(AUC)
            if epoch % 10 == 0:
                print(f'Pred Epoch {epoch} — train: {train_loss:.4f}  valid: {vali_loss:.4f}  AUC: {AUC:.4f}')

        if i == iters - 1:
            for data, name in [(AUC_list, 'AUC'), (loss_list, 'Loss'), (vali_loss_list, 'ValidLoss')]:
                plt.plot(data)
                plt.title(f'{name} vs Epoch')
                plt.savefig(f'{name}plot_topo_KAN.png')
                plt.show()

        All_AUC.append(max(AUC_list))
        print('Iteration done!')

    print(All_AUC)
    print("mean:", statistics.mean(All_AUC))
    if iters > 1:
        print("std:", statistics.stdev(All_AUC))