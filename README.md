# CKANGA - Contrastive Kolmogorov Arnold Network Graph Autoencoder

<img width="1555" height="673" alt="CKANGA" src="https://github.com/user-attachments/assets/7fafc77b-b502-4cf4-a299-572727c6b779" />


## data
Contains the raw and processed MoleculeNet datasets, with accompanying R cleaning script, 
as well as the subsets of the ChEMBL small molecules database, also with accompanying R script. 
The original CheEMBL small organic csv file is not included, but can be found here https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

## graph_preprocessing
Contains the scripts for feature construction of the molecular graph from the SMILES representations.
mol_processing.py is called to generate for each dataset.

## models
Contains the overarching model scripts for the KAN and the MLP. Model components are reused
in other scripts, but are typically lightly modified versions of these.  

## parameter_tuning
Contains the hyperparameter tuning scripts, in the order that they were run. The results of the
tuning script are also included

## encoder_training
Includes the latent training sciprts for each of the models. The KAN scripts were overwritten to produce
KAN modified and KAN mini, but can be altered according to needs. The parameters of the trained models 
are also included for reproducibility.

## model_testing
Includes the testing scripts that iterate over the datasets using the trained model parameters. 
All models call Model_testing.py, which must be modified to accept a new model in the 'if' statements
where the encoder is instantiated

## results
Contains Json outputs of the model results. Summary.py will produce a nice summary table to the 
command line when run. Files are labeled '{dataset}{model}_{encoder train size}.json'.


There is also an archive for old figures and scripts used in initial project scoping and testing of 
the KAN networks.

