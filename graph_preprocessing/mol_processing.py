import graph_processing
import numpy as np

import os
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
from torch_geometric.data.data import DataEdgeAttr  
import resource

if __name__ == '__main__':
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    model = 'GAE_MLP'
    train, test, valid = 1,0,0  

    #Base State
    batches=512
    #decoder_layers = 
    #dec_epochs = 
    #learn_rate = 
    #latent_size = 
    #readout_layers = 
    #hidden_width = 
    #message_layers = 
    #topo_ratio =
    #cont_weight =
    #margin = 

    seed = 23
    graph_processing.graph_processing('small_molecules_200k',batches,1,0,0)
    graph_processing.graph_processing('small_molecules_500k',batches,1,0,0)
    graph_processing.graph_processing('small_molecules_1m',batches,1,0,0)
    '''
    GAEMLP_latent.GAEMLP_latent_train(f'{dataset}_{batches}',
                                        learn_rate,
                                        dec_epochs,
                                        message_layers,
                                        readout_layers, 
                                        decoder_layers,
                                        hidden_width,
                                        latent_size, 
                                        topo_weight=topo_ratio,
                                        eval_every=5,patience=30,prediction_model=None,
                                        pred_epochs=0, seed = seed,
                                        contrastive_weight=cont_weight,
                                        margin=margin)
                                            '''

