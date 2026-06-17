import Model_Testing
import os
import json
import graph_processing

model = 'GAEMLP_trained_encoder_200k.pth'
name = model.split('.')[0]
latent_size = 128
if not os.path.isfile('bace'+name+'.json'):
    graph_processing.graph_processing('bace',128,0.8,0.1,0.1)
    bace = Model_Testing.Testing_Script('bace',128,latent_size,model)
    with open('bace'+name+'.json', 'w') as f:
        json.dump({'bace'+name: bace}, f, indent=4)

if not os.path.isfile('bbbp'+name+'.json'):
    graph_processing.graph_processing('bbbp',128,0.8,0.1,0.1)
    bbbp = Model_Testing.Testing_Script('bbbp',128,latent_size,model)
    with open('bbbp'+name+'.json', 'w') as f:
        json.dump({'bbbp'+name: bbbp}, f, indent=4)

if not os.path.isfile('hiv'+name+'.json'):
    graph_processing.graph_processing('hiv',256,0.8,0.1,0.1)
    hiv = Model_Testing.Testing_Script('hiv',256,latent_size,model)
    with open('hiv'+name+'.json', 'w') as f:
        json.dump({'hiv'+name: hiv}, f, indent=4)

if not os.path.isfile('tox21'+name+'.json'):
    graph_processing.graph_processing('tox21',256,0.8,0.1,0.1)
    tox21 = Model_Testing.Testing_Script('tox21',256,latent_size,model)
    with open('tox21'+name+'.json', 'w') as f:
        json.dump({'tox21'+name: tox21}, f, indent=4)


model = 'GAEMLP_trained_encoder_500k.pth'
name = model.split('.')[0]
latent_size = 128
if not os.path.isfile('bace'+name+'.json'):
    graph_processing.graph_processing('bace',128,0.8,0.1,0.1)
    bace = Model_Testing.Testing_Script('bace',128,latent_size,model)
    with open('bace'+name+'.json', 'w') as f:
        json.dump({'bace'+name: bace}, f, indent=4)

if not os.path.isfile('bbbp'+name+'.json'):
    graph_processing.graph_processing('bbbp',128,0.8,0.1,0.1)
    bbbp = Model_Testing.Testing_Script('bbbp',128,latent_size,model)
    with open('bbbp'+name+'.json', 'w') as f:
        json.dump({'bbbp'+name: bbbp}, f, indent=4)

if not os.path.isfile('hiv'+name+'.json'):
    graph_processing.graph_processing('hiv',256,0.8,0.1,0.1)
    hiv = Model_Testing.Testing_Script('hiv',256,latent_size,model)
    with open('hiv'+name+'.json', 'w') as f:
        json.dump({'hiv'+name: hiv}, f, indent=4)

if not os.path.isfile('tox21'+name+'.json'):
    graph_processing.graph_processing('tox21',256,0.8,0.1,0.1)
    tox21 = Model_Testing.Testing_Script('tox21',256,latent_size,model)
    with open('tox21'+name+'.json', 'w') as f:
        json.dump({'tox21'+name: tox21}, f, indent=4)

model = 'GAEMLP_trained_encoder_1m.pth'
name = model.split('.')[0]
latent_size = 128
if not os.path.isfile('bace'+name+'.json'):
    graph_processing.graph_processing('bace',128,0.8,0.1,0.1)
    bace = Model_Testing.Testing_Script('bace',128,latent_size,model)
    with open('bace'+name+'.json', 'w') as f:
        json.dump({'bace'+name: bace}, f, indent=4)

if not os.path.isfile('bbbp'+name+'.json'):
    graph_processing.graph_processing('bbbp',128,0.8,0.1,0.1)
    bbbp = Model_Testing.Testing_Script('bbbp',128,latent_size,model)
    with open('bbbp'+name+'.json', 'w') as f:
        json.dump({'bbbp'+name: bbbp}, f, indent=4)

if not os.path.isfile('hiv'+name+'.json'):
    graph_processing.graph_processing('hiv',256,0.8,0.1,0.1)
    hiv = Model_Testing.Testing_Script('hiv',256,latent_size,model)
    with open('hiv'+name+'.json', 'w') as f:
        json.dump({'hiv'+name: hiv}, f, indent=4)

if not os.path.isfile('tox21'+name+'.json'):
    graph_processing.graph_processing('tox21',256,0.8,0.1,0.1)
    tox21 = Model_Testing.Testing_Script('tox21',256,latent_size,model)
    with open('tox21'+name+'.json', 'w') as f:
        json.dump({'tox21'+name: tox21}, f, indent=4)
