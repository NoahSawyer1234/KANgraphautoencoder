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

import GAE_KAN
import GAE_MLP
import GCN_KAN
import GCN_MLP
import graph_processing
import numpy as np
import json

if __name__ == '__main__':
    batches = 128
    harmonics = 1
    learn_rate = 0.0001
    epochs = 500
    hidden_width = 16
    latent_size = 128
    message_layers = 2
    readout_layers = 1
    decoder_layers = 1
    pred_layers = 1
    iterations = 1
    model = 'GAE_MLP'
    dataset = 'bace'
    print('Run starting...')

    graph_processing.graph_processing(dataset,batches,0.8,0,0.2)
    res = GAE_MLP.GAE_MLP_Script(batches,dataset + f'_{batches}',iterations,learn_rate,epochs,epochs,
                                 message_layers,readout_layers,pred_layers,decoder_layers,hidden_width,latent_size)
    '''    res = GAE_KAN.GAE_KAN_Script(batches,dataset + f'_{batches}',iterations,learn_rate,epochs,epochs,harmonics,
                                 message_layers,readout_layers,pred_layers,decoder_layers,hidden_width,latent_size)'''
    print(res)
    print(np.mean(res))

    with open('test.json', 'w') as f:
        json.dump(res, f, indent=4)
    
