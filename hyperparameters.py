import GCN_KAN
import GCN_MLP
import GAE_KAN
import GAE_MLP
import graph_processing
import numpy as np
import json

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

if __name__ == '__main__':
    #GCN/general features
    batches = [128,256,512]
    harmonics = [1,2,3,4]
    learn_rate = [0.001,0.0001,0.00001]
    epochs = [250,500,1000,2000]
    hidden_width = [32,64,128]
    message_layers = [1,2,3,4,5]
    iterations = 50

    #AE features
    latent_size = [64,128,256,512]
    pred_layers = 2
    enc_epochs = [250,500,1000,2000]

    model = 'GCN_KAN'
    dataset = 'bace'
    train, test, valid = 0.8,0,0.2 ###### Need to comment out train parts of Architectures to run 
    best_max_auc = 0
    best_hyperparams = ""
    architecture = model.split('_')[0]

    if architecture == 'GCN':
        for b in batches:
            graph_processing.graph_processing(dataset,b,train,test,valid)
            file_name = dataset + f'_{b}'
            if model == 'GCN_KAN':
                for h in harmonics:
                    for lr in learn_rate:
                        for e in epochs:
                            for m in message_layers:
                                for hw in hidden_width:
                                    max_auc_list = GCN_KAN.GCN_KAN_Script(b,file_name,iterations,lr,e,h,m,hw)
                                    if np.mean(best_max_auc) > best_max_auc:
                                        best_max_auc = np.mean(best_max_auc)
                                        best_hyperparams = [b,h,lr,e,m,hw]
            elif model == 'GCN_MLP':
                for lr in learn_rate:
                    for e in epochs:
                        for m in message_layers:
                            for hw in hidden_width:
                                max_auc_list = GCN_MLP.GCN_MLP_Script(b,file_name,iterations,lr,e,m,hw)
                                if np.mean(best_max_auc) > best_max_auc:
                                    best_max_auc = np.mean(best_max_auc)
                                    best_hyperparams = [b,'NA',lr,e,m,hw]
        best = {
            "batch_size": best_hyperparams[0],
            "num_harmonics": best_hyperparams[1],
            "learning_rate": best_hyperparams[2],
            "num_epochs": best_hyperparams[3],
            "message_layers": best_hyperparams[4],
            "hidden_width": best_hyperparams[5],
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
                                for hw in hidden_width:
                                    for ee in enc_epochs:
                                        for pl in pred_layers:
                                            for ls in latent_size:
                                                max_auc_list = GAE_KAN.GAE_KAN_Script(b,file_name,
                                                                                      iterations,lr,
                                                                                      e,ee,h,m,pl,
                                                                                      hw,ls)
                                                if np.mean(best_max_auc) > best_max_auc:
                                                    best_max_auc = np.mean(best_max_auc)
                                                    best_hyperparams = [b,h,lr,e,m,hw,ee,pl,ls]
            if model == 'GAE_MLP':
                for lr in learn_rate:
                    for e in epochs:
                        for m in message_layers:
                            for hw in hidden_width:
                                for ee in enc_epochs:
                                    for pl in pred_layers:
                                        for ls in latent_size:
                                            max_auc_list = GAE_MLP.GAE_MLP_Script(b,file_name,
                                                                                    iterations,lr,
                                                                                    e,ee,m,pl,
                                                                                    hw,ls)
                                            if np.mean(best_max_auc) > best_max_auc:
                                                best_max_auc = np.mean(best_max_auc)
                                                best_hyperparams = [b,h,lr,e,m,hw,ee,pl,ls]

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
        }
        with open(f'{model}_best.json', 'w') as f:
            json.dump(best, f, indent=4)






