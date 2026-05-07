import GAE_KAN
import graph_processing
import numpy as np
import json
import optimised_GAEKAN

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
    #base_state:
    #Tested and chosen
    batches = 128
    #Yet to choose
    harmonics = 1
    learn_rate = 0.0001
    epochs = 500
    hidden_width = 64
    latent_size = 128
    message_layers = 3
    readout_layers = 1
    decoder_layers = 1

    #For prediction layer
    pred_module = KA_latentpred(latent_size,64,1,1,2,True)

    iters = 50
    model = 'GAE_MLP'
    dataset = 'bace'
    results = {}
    for d in [1,2,3]:
        graph_processing.graph_processing(dataset,batches,0.8,0,0.2)
        max_auc_list = optimised_GAEKAN.GAE_KAN_Script(f'{dataset}_{batches}',iters, learn_rate,epochs,harmonics,message_layers,
                                                       readout_layers,d,hidden_width,latent_size,
                                                       eval_every=5,patience=30,prediction_model=pred_module,pred_epochs=500)
        results[f'{d}'] = max_auc_list
        results[f'{d}_mean'] = np.mean(max_auc_list)
    with open('dec_layers_tuning.json', 'w') as f:
        json.dump(results, f, indent=4)


    '''
        if architecture == 'GCN':
            for b in batches:
                graph_processing.graph_processing(dataset,b,train,test,valid)
                file_name = dataset + f'_{b}'
                if model == 'GCN_KAN':
                    for h in harmonics:
                        for lr in learn_rate:
                            for e in epochs:
                                for m in message_layers:
                                    for r in readout_layers:
                                        for hw in hidden_width:
                                            max_auc_list = GCN_KAN.GCN_KAN_Script(b,file_name,iterations,lr,e,h,m,r,hw)
                                            if np.mean(best_max_auc) > best_max_auc:
                                                best_max_auc = np.mean(best_max_auc)
                                                best_hyperparams = [b,h,lr,e,m,r,hw]
                elif model == 'GCN_MLP':
                    for lr in learn_rate:
                        for e in epochs:
                            for m in message_layers:
                                for r in readout_layers:
                                    for hw in hidden_width:
                                        max_auc_list = GCN_MLP.GCN_MLP_Script(b,file_name,iterations,lr,e,m,r,hw)
                                        if np.mean(best_max_auc) > best_max_auc:
                                            best_max_auc = np.mean(best_max_auc)
                                            best_hyperparams = [b,'NA',lr,e,m,r,hw]
            best = {
                "batch_size": best_hyperparams[0],
                "num_harmonics": best_hyperparams[1],
                "learning_rate": best_hyperparams[2],
                "num_epochs": best_hyperparams[3],
                "message_layers": best_hyperparams[4],
                "readout_layers": best_hyperparams[5],
                "hidden_width": best_hyperparams[6]
            }
            with open(f'{model}_best.json', 'w') as f:
                json.dump(best, f, indent=4)

        if architecture == 'GAE':
            for b in batches:
                graph_processing.graph_processing(dataset,b,train,test,valid)
                file_name = dataset + f'_{b}'
                if model == 'GAE_KAN':
                    for h in harmonics:
                        for lr in learn_rate:
                            for e in epochs:
                                for m in message_layers:
                                    for r in readout_layers:
                                        for hw in hidden_width:
                                            for ee in enc_epochs:
                                                for pl in pred_layers:
                                                    for d in dec_layers:
                                                        for ls in latent_size:
                                                            max_auc_list = GAE_KAN.GAE_KAN_Script(b,file_name,
                                                                                                iterations,lr,
                                                                                                e,ee,h,m,r,pl,
                                                                                                d,hw,ls)
                                                            if np.mean(best_max_auc) > best_max_auc:
                                                                best_max_auc = np.mean(best_max_auc)
                                                                best_hyperparams = [b,h,lr,e,m,hw,ee,pl,ls]
                if model == 'GAE_MLP':
                    for lr in learn_rate:
                        for e in epochs:
                            for m in message_layers:
                                for r in readout_layers:
                                    for hw in hidden_width:
                                        for ee in enc_epochs:
                                            for pl in pred_layers:
                                                for d in dec_layers:
                                                    for ls in latent_size:
                                                        max_auc_list = GAE_MLP.GAE_MLP_Script(b,file_name,
                                                                                                iterations,lr,
                                                                                                e,ee,m,r,pl,
                                                                                                d,hw,ls)
                                                        if np.mean(best_max_auc) > best_max_auc:
                                                            best_max_auc = np.mean(best_max_auc)
                                                            best_hyperparams = [b,h,lr,e,m,hw,ee,pl,ls,d,r]

            best = {
                "batch_size": best_hyperparams[0],
                "num_harmonics": best_hyperparams[1],
                "learning_rate": best_hyperparams[2],
                "num_epochs": best_hyperparams[3],
                "message_layers": best_hyperparams[4],
                "hidden_width": best_hyperparams[5],
                "encoding_epochs": best_hyperparams[6],
                "prediction_layers": best_hyperparams[7],
                "latent_size": best_hyperparams[8],
                "decoder_layers": best_hyperparams[9],
                "readout_layers": best_hyperparams[10]

            }
            with open(f'{model}_best.json', 'w') as f:
                json.dump(best, f, indent=4)

    '''




