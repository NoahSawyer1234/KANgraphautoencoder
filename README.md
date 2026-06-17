# CKANGA - Contrastive Kolmogorov Arnold Network Graph Autoencoder

<img width="1555" height="673" alt="CKANGA" src="https://github.com/user-attachments/assets/7fafc77b-b502-4cf4-a299-572727c6b779" />


### data
Contains the raw and processed MoleculeNet datasets, with accompanying R cleaning script, 
as well as the subsets of the ChEMBL small molecules database, also with accompanying R script. 
The original CheEMBL small organic csv file is not included, but can be found here https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

### graph_preprocessing
Contains the scripts for feature construction of the molecular graph from the SMILES representations.
mol_processing.py is called to generate for each dataset.

### models
Contains the overarching model scripts for the KAN and the MLP. Model components are reused
in other scripts, but are typically lightly modified versions of these.  

### parameter_tuning
Contains the hyperparameter tuning scripts, in the order that they were run. The results of the
tuning script are also included

### encoder_training
Includes the latent training sciprts for each of the models. The KAN scripts were overwritten to produce
KAN modified and KAN mini, but can be altered according to needs. The parameters of the trained models 
are also included for reproducibility.

### model_testing
Includes the testing scripts that iterate over the datasets using the trained model parameters. 
All models call Model_testing.py, which must be modified to accept a new model in the 'if' statements
where the encoder is instantiated

### results
Contains Json outputs of the model results. Summary.py will produce a nice summary table to the 
command line when run. Files are labeled '{dataset}{model}_{encoder train size}.json'.


There is also an archive for old figures and scripts used in initial project scoping and testing of 
the KAN networks.

# Encoder Benchmark Results
 
AUC-ROC across MoleculeNet datasets (n = 100 runs per row). Bold = best mean AUC for that dataset across all models.
** Note for literature comparisons - All models reported using Scaffold Split, which is standard for BACE, BBBP and HIV. For Tox21, reported results are commonly random splits, which generally have greater performance.
 
| Model / Dataset       | Mean   | Std    | Min    | Max    | <0.5 | n   |
|------------------------|-------:|-------:|-------:|-------:|----:|----:|
| **KAN 200K**           |        |        |        |        |     |     |
| BACE                    | 0.5564 | 0.0209 | 0.5159 | 0.6667 | 0   | 100 |
| BBBP                    | 0.5242 | 0.0324 | 0.4328 | 0.5999 | 20  | 100 |
| HIV                     | 0.5476 | 0.0218 | 0.4953 | 0.5909 | 2   | 100 |
| TOX21                   | 0.5884 | 0.0069 | 0.5680 | 0.6060 | 0   | 100 |
| **KAN 500K**           |        |        |        |        |     |     |
| BACE                    | 0.4876 | 0.0403 | 0.4097 | 0.6056 | 68  | 100 |
| BBBP                    | 0.5160 | 0.0391 | 0.4468 | 0.6062 | 43  | 100 |
| HIV                     | 0.5241 | 0.0191 | 0.4891 | 0.5763 | 9   | 100 |
| TOX21                   | 0.5926 | 0.0080 | 0.5763 | 0.6122 | 0   | 100 |
| **KAN 1M**             |        |        |        |        |     |     |
| BACE                    | 0.5401 | 0.0494 | 0.4374 | 0.6533 | 23  | 100 |
| BBBP                    | 0.5269 | 0.0380 | 0.4453 | 0.6463 | 28  | 100 |
| HIV                     | 0.5316 | 0.0192 | 0.4973 | 0.5855 | 1   | 100 |
| TOX21                   | 0.5945 | 0.0086 | 0.5736 | 0.6116 | 0   | 100 |
| **KAN Modified 200K**  |        |        |        |        |     |     |
| BACE                    | 0.6418 | 0.0249 | 0.5789 | 0.6963 | 0   | 100 |
| BBBP                    | 0.5949 | 0.0350 | 0.4837 | 0.7205 | 1   | 100 |
| HIV                     | 0.6812 | 0.0150 | 0.6535 | 0.7210 | 0   | 100 |
| TOX21                   | 0.5954 | 0.0228 | 0.5368 | 0.6470 | 0   | 100 |
| **KAN Modified 500K**  |        |        |        |        |     |     |
| BACE                    | 0.6707 | 0.0521 | 0.2992 | 0.7087 | 2   | 100 |
| BBBP                    | 0.5897 | 0.0111 | 0.5787 | 0.6694 | 0   | 100 |
| HIV                     | 0.6213 | 0.0040 | 0.6165 | 0.6363 | 0   | 100 |
| TOX21                   | 0.5506 | 0.0279 | 0.4977 | 0.6180 | 1   | 100 |
| **KAN Modified 1M**    |        |        |        |        |     |     |
| BACE                    | 0.6799 | 0.0071 | 0.6684 | 0.7067 | 0   | 100 |
| BBBP                    | 0.5999 | 0.0053 | 0.5886 | 0.6151 | 0   | 100 |
| HIV                     | 0.6509 | 0.0018 | 0.6491 | 0.6636 | 0   | 100 |
| TOX21                   | 0.5357 | 0.0230 | 0.4977 | 0.6028 | 1   | 100 |
| **KAN Mini 200K**      |        |        |        |        |     |     |
| BACE                    | 0.7624 | 0.0125 | 0.7376 | 0.7969 | 0   | 100 |
| BBBP                    | **0.7594** | 0.0099 | 0.7305 | 0.7910 | 0   | 100 |
| HIV                     | **0.7314** | 0.0128 | 0.7097 | 0.7776 | 0   | 100 |
| TOX21                   | **0.7481** | 0.0036 | 0.7402 | 0.7591 | 0   | 100 |
| **KAN Mini 500K**      |        |        |        |        |     |     |
| BACE                    | 0.7917 | 0.0111 | 0.7650 | 0.8209 | 0   | 100 |
| BBBP                    | 0.7301 | 0.0113 | 0.7060 | 0.7713 | 0   | 100 |
| HIV                     | 0.7189 | 0.0087 | 0.6990 | 0.7395 | 0   | 100 |
| TOX21                   | 0.7450 | 0.0040 | 0.7342 | 0.7533 | 0   | 100 |
| **KAN Mini 1M**        |        |        |        |        |     |     |
| BACE                    | **0.8114** | 0.0105 | 0.7848 | 0.8349 | 0   | 100 |
| BBBP                    | 0.7383 | 0.0114 | 0.7084 | 0.7772 | 0   | 100 |
| HIV                     | 0.7300 | 0.0087 | 0.7083 | 0.7631 | 0   | 100 |
| TOX21                   | 0.7310 | 0.0058 | 0.7140 | 0.7457 | 0   | 100 |
| **MLP 200K**           |        |        |        |        |     |     |
| BACE                    | 0.7281 | 0.0067 | 0.7171 | 0.7509 | 0   | 100 |
| BBBP                    | 0.6328 | 0.0043 | 0.6264 | 0.6485 | 0   | 100 |
| HIV                     | 0.6748 | 0.0114 | 0.6521 | 0.7042 | 0   | 100 |
| TOX21                   | 0.7221 | 0.0050 | 0.7092 | 0.7344 | 0   | 100 |
| **MLP 500K**           |        |        |        |        |     |     |
| BACE                    | 0.7222 | 0.0134 | 0.6904 | 0.7554 | 0   | 100 |
| BBBP                    | 0.7399 | 0.0097 | 0.7170 | 0.7674 | 0   | 100 |
| HIV                     | 0.6450 | 0.0125 | 0.6061 | 0.6690 | 0   | 100 |
| TOX21                   | 0.7200 | 0.0053 | 0.7089 | 0.7323 | 0   | 100 |
| **MLP 1M**             |        |        |        |        |     |     |
| BACE                    | 0.7151 | 0.0090 | 0.6936 | 0.7359 | 0   | 100 |
| BBBP                    | 0.7304 | 0.0136 | 0.6976 | 0.7583 | 0   | 100 |
| HIV                     | 0.7091 | 0.0124 | 0.6762 | 0.7418 | 0   | 100 |
| TOX21                   | 0.7173 | 0.0042 | 0.7093 | 0.7289 | 0   | 100 |
