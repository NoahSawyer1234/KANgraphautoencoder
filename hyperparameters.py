import GCN_KAN
import GCN_MLP
import graph_processing
import numpy as np
import json

if __name__=='main':
    batches = [128,256,512]
    harmonics = [1,2,3,4]
    learn_rate = [0.001,0.0001,0.0001]
    epochs = [250,500,1000,2000]
    hidden_width = [32,64,128]
    message_layers = [1,2,3,4,5]
    iterations = 50
    model = 'GCN_KAN'
    dataset = 'bace'
    train, test, valid = 0.8,0,0.2
    best_max_auc = 0
    best_hyperparams = ""

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






