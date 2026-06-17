import optimised_GAEKAN
import torch.nn as nn
import torch
import os
import numpy as np
import json

decoder_layers = 3
dec_epochs = 500
learn_rate = 10**-4
latent_size = 512
readout_layers = 3
hidden_width = 256
message_layers = 4
topo_ratio= 0.3
harmonics = 2
topo_ratio = 0.3    
cont_weight = 0.1
margin = 4

tune_iters = 20
test_iters = 100

width = [32,64,128,256]
depth = [2,3,4]
lr = [10**-3,10**-4,10**-5]

class MLP_latentpred(nn.Module):
    def __init__(self, latent_feat, hidden_feat, out_feat, p_num_layers, use_bias=True):
        super().__init__()
        if p_num_layers == 1:
            layers = [nn.Linear(latent_feat,out_feat,bias=use_bias)]
        else:
            layers = [nn.Linear(latent_feat,hidden_feat,bias=use_bias),
                      nn.ReLU()]
            for _ in range(p_num_layers - 2):
                layers.append(nn.Linear(hidden_feat,hidden_feat,bias=use_bias))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_feat,out_feat,bias=use_bias))
        self.predictor = nn.Sequential(*layers)
 
    def forward(self, latent):
        return self.predictor(latent) 
    

def tester(dataset, batches):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'running on GPU: {torch.cuda.get_device_name(0)}')
        n_workers = min(4, os.cpu_count() or 1)
        pin_mem   = True
    else:
        device = torch.device('cpu')
        print('running on CPU')
        torch.set_num_threads(os.cpu_count() or 1)
        n_workers = min(8, os.cpu_count() or 1)
        pin_mem   = False
    best_combo_auc =0
    best_combo = []
    target_dim = 12 if dataset == 'tox21' else 1
    for w in width:
        for d in depth:
            all_AUC = []
            print("On model ",w,d)
            pred_module = MLP_latentpred(latent_size,w,target_dim,d,True).to(device)
            max_auc_list = optimised_GAEKAN.GAE_KAN_Script(f'{dataset}_{batches}',
                                                        tune_iters, 
                                                        learn_rate,
                                                        dec_epochs,
                                                        harmonics,
                                                        message_layers,
                                                        readout_layers, 
                                                        decoder_layers,
                                                        hidden_width,
                                                        latent_size, 
                                                        topo_weight=topo_ratio,
                                                        eval_every=5,patience=30,prediction_model=pred_module,
                                                        pred_epochs=1000, seed = 0,
                                                        contrastive_weight=cont_weight,
                                                        margin=margin)
            if np.mean(max_auc_list)>best_combo_auc:
                best_combo = [w,d]
                best_combo_auc = np.mean(max_auc_list)
    pred_module = MLP_latentpred(latent_size,w,target_dim,d,True).to(device)
    test_auc = optimised_GAEKAN.GAE_KAN_Script(f'{dataset}_{batches}',
                                            test_iters, 
                                            learn_rate,
                                            dec_epochs,
                                            harmonics,
                                            message_layers,
                                            readout_layers, 
                                            decoder_layers,
                                            hidden_width,
                                            latent_size, 
                                            topo_weight=topo_ratio,
                                            eval_every=5,patience=30,prediction_model=pred_module,
                                            pred_epochs=1000, seed = 0,
                                            contrastive_weight=cont_weight,
                                            margin=margin)
    return test_auc

if not os.path.isfile('bace_only_KAN.json'):
    bace = tester('bace',128)
    with open('bace_only_KAN.json', 'w') as f:
        json.dump({'bace_results': bace}, f, indent=4)

if not os.path.isfile('bbbp_only_KAN.json'):
    bbbp = tester('bbbp',128)
    with open('bbbp_only_KAN.json', 'w') as f:
        json.dump({'bbbp_results': bbbp}, f, indent=4)

if not os.path.isfile('hiv_only_KAN.json'):
    hiv = tester('hiv',256)
    with open('hiv_only_KAN.json', 'w') as f:
        json.dump({'hiv_results': hiv}, f, indent=4)

if not os.path.isfile('tox21_only_KAN.json'):
    tox21 = tester('tox21',256)
    with open('tox21_only_KAN.json', 'w') as f:
        json.dump({'tox21_results': tox21}, f, indent=4)

    