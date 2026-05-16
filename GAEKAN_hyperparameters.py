import graph_processing
import numpy as np
import json
import optimised_GAEKAN

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

# Need this for the prediction module
class KA_latentpred(nn.Module):
    def __init__(self, latent_feat, hidden_feat, out_feat, num_harmonics, p_num_layers, use_bias=True):
        super().__init__()
        if p_num_layers == 1:
            layers = [optimised_GAEKAN.KAN_node_embedding(latent_feat, out_feat, num_harmonics, addbias=use_bias)]
        else:
            layers = [optimised_GAEKAN.KAN_node_embedding(latent_feat, hidden_feat, num_harmonics, addbias=use_bias)]
            for _ in range(p_num_layers - 2):
                layers.append(optimised_GAEKAN.KAN_node_embedding(hidden_feat, hidden_feat, num_harmonics, addbias=use_bias))
            layers.append(optimised_GAEKAN.KAN_node_embedding(hidden_feat, out_feat, num_harmonics, addbias=use_bias))
        layers.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*layers)
 
    def forward(self, latent):
        return self.predictor(latent)

if __name__ == '__main__':
    model = 'GAE_KAN'
    dataset = 'bace'
    train, test, valid = 0.8,0,0.2  
    auc_list = []

    #Tested and chosen (in order from top to bottom)
    batches=128
    decoder_layers = [1]
    dec_epochs = 500
    learn_rate = [10**-4]
    latent_size = [128]
    readout_layers = [1]
    hidden_width = [64]
    message_layers = [1]
    topo_ratio=[0.5]
    harmonics = [1]
    topo_ratio = [0.5]    
    cont_weight = [0.1]
    margin = [1]

    iters = 50
    seed = 23
    
      #Nice function that I can keep running until timeout on katana instead of repeatedly uploading!!
    def stage_tuner(name,parameter,values):
        results = {}
        if not os.path.isfile(name):
            for val in values:
                parameter[0] = val
                optimised_GAEKAN.set_seed(seed)
                graph_processing.graph_processing(dataset,batches,0.8,0,0.2)
                pred_module = KA_latentpred(latent_size[0],64,1,1,2,True)
                max_auc_list = optimised_GAEKAN.GAE_KAN_Script(f'{dataset}_{batches}',
                                                                iters, 
                                                                learn_rate[0],
                                                                dec_epochs,
                                                                harmonics[0],
                                                                message_layers[0],
                                                                readout_layers[0], 
                                                                decoder_layers[0],
                                                                hidden_width[0],
                                                                latent_size[0], 
                                                                topo_weight=topo_ratio[0],
                                                                eval_every=5,patience=30,prediction_model=pred_module,
                                                                pred_epochs=500, seed = seed,
                                                                contrastive_weight=cont_weight,
                                                                margin=margin)
                results[f'{val}'] = max_auc_list
                results[f'{val}_mean'] = np.mean(max_auc_list)
            with open(name, 'w') as f:
                json.dump(results, f, indent=4)

        with open(name, 'r') as f:
            data = json.load(f)
        best = 0
        for val in values:
            if data[f'{val}_mean'] >best:
                best = data[f'{val}_mean'] 
                parameter[0] = val
    stage_tuner("contrastive_tuning_MLP.json",cont_weight,[0,0.1,0.5,1,2])
    stage_tuner("margin_tuning_MLP.json",margin,[0.1,0.5,1,5,10])
    stage_tuner("learn_rate_tuning_KAN.json",learn_rate,[10**-3,10**-4,10**-5])
    stage_tuner("dec_layer_tuning_KAN.json",decoder_layers,[1,2,3,4])
    stage_tuner("hidden_width_tuning_KAN.json",hidden_width,[16,32,64,128,256])
    stage_tuner("harmonics_tuning_KAN.json",harmonics,[1,2,3,4,5])
    stage_tuner("readout_layers_tuning_KAN.json",readout_layers,[1,2,3,4,5])
    stage_tuner("message_layers_tuning_KAN.json",hidden_width,[1,2,3,4,5])
    stage_tuner("latent_size_tuning_KAN.json",latent_size,[32,64,128,256,512])
    stage_tuner("topo_ratio_tuning_KAN.json",topo_ratio,[0.1,0.3,0.5,0.7,0.9])




