#####   Importing  ######
#   - PyTorch for network architecture
#   - DGL for graph network data structuring
#   - Jarvis for chemical attribute production
import dgl
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from jarvis.core.specie import chem_data, get_node_attributes
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd

####    Function for bond (edge) length  #####
def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis

#Now 18 total, 15 bonded, 5 non-bonded       
############ Removed repeat distance variables, and added exact bond length instead of estimate

## CHNAGED BOND  LENGTH VARIABLES
def encode_bond_21(bond,conf):
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1
    
    bond_type = [0] * 4
    bond_type[int(bond.GetBondTypeAsDouble()) - 1] = 1
    
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    bond_length = Chem.rdMolTransforms.GetBondLength(conf, idx1, idx2)
    
    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1
    
    non_bond_feature = [0]*5

    edge_encode = bond_dir + bond_type + [1/bond_length**2,bond_length] + in_ring + non_bond_feature

    return edge_encode
 
# 6 var now
def non_bonded(charge_list,i,j,dis):
    charge_list = [float(charge) for charge in charge_list] 
    q_i = [charge_list[i]]
    q_j = [charge_list[j]]
    q_ij = [charge_list[i]*charge_list[j]]
    dis_1 = [dis]
    dis_2 = [1/dis**2]

    return q_i + q_j + q_ij + dis_1 + dis_2 


# Tries to tell if there's a merck force field available (coordinates)
def mmff_force_field(mol):
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return True
    except ValueError:
        return False

def check_common_elements(list1, list2, element1, element2):
    if len(list1) != len(list2):
        return False  
    for i in range(len(list1)):
        if list1[i] == element1 and list2[i] == element2:
            return True  
    return False  

def atom_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    else:
        mol = Chem.AddHs(mol) 
    sps_features = []
    coor = []
    edge_id = []
    atom_charges = []
    
    smiles_with_hydrogens = Chem.MolToSmiles(mol)

    tmp = []
    for num in smiles_with_hydrogens:
        if num not in ['[',']','(',')']:
            tmp.append(num)

    sm = {}
    for atom in mol.GetAtoms():
        atom_index = atom.GetIdx()
        sm[atom_index] =  atom.GetSymbol()

    ### Upper barrier for molecule size
    Num_atoms = len(tmp)
    if Num_atoms > 700:
        g = False

    else:
        if mmff_force_field(mol) == True:
            num_conformers = mol.GetNumConformers()
            if num_conformers > 0:
                AllChem.ComputeGasteigerCharges(mol)
                conf = mol.GetConformer()
                for ii, s in enumerate(mol.GetAtoms()):
                    per_atom_feat = []
                    # Gets node attributes using cgcnn in our case, 92-dim
                    feat = list(get_node_attributes(s.GetSymbol(), atom_features='cgcnn'))
                    per_atom_feat.extend(feat)

                    sps_features.append(per_atom_feat )
                        
                    pos = mol.GetConformer().GetAtomPosition(ii)
                    coor.append([pos.x, pos.y, pos.z])

                    charge = s.GetProp("_GasteigerCharge")
                    atom_charges.append(charge)

                edge_features = []
                src_list, dst_list = [], []
                for bond in mol.GetBonds():
                    src = bond.GetBeginAtomIdx()
                    dst = bond.GetEndAtomIdx()
                    src_list.append(src)
                    src_list.append(dst)
                    dst_list.append(dst)
                    dst_list.append(src)

                    per_bond_feat = []
                                       
                    per_bond_feat.extend(encode_bond_21(bond,conf))
                  
                    edge_features.append(per_bond_feat)
                    edge_features.append(per_bond_feat)
                    edge_id.append([1])
                    edge_id.append([1])

                for i in range(len(coor)):
                    coor_i =  np.array(coor[i])
                    for j in range(i+1, len(coor)):
                        coor_j = np.array(coor[j])
                        s_d_dis = calculate_dis(coor_i,coor_j)
                        if s_d_dis <= 5:
                            if check_common_elements(src_list,dst_list,i,j):
                                src_list.append(i)
                                src_list.append(j)
                                dst_list.append(j)
                                dst_list.append(i)
                                per_bond_feat = [0]*15
                                per_bond_feat.extend(non_bonded(atom_charges,i,j,s_d_dis))

                                edge_features.append(per_bond_feat)
                                edge_features.append(per_bond_feat)
                                edge_id.append([0])
                                edge_id.append([0])

                coor_tensor = torch.tensor(coor, dtype=torch.float32)
                edge_feats = torch.tensor(edge_features, dtype=torch.float32)
                edge_id_feats = torch.tensor(edge_id, dtype=torch.float32)

                node_feats = torch.tensor(sps_features,dtype=torch.float32)

                
                # Number of atoms
                num_atoms = mol.GetNumAtoms()

                # Create a graph. undirected_graph
                g = dgl.DGLGraph()
                g.add_nodes(num_atoms)
                g.add_edges(src_list, dst_list)
                
                g.ndata['feat'] = node_feats
                g.ndata['coor'] = coor_tensor  
                g.edata['feat'] = edge_feats
                g.edata['id'] = edge_id_feats
            
            else:
                g = False
        else:
            g = False
    return g


def path_complex_mol(Smile):
    g = atom_to_graph(Smile)
    if g != False:
        return g
    else:
        return False
    



##### DATASET PROCESSING CLASSES
class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test 
    subsets.
    """
    def __init__(self):
        super(Splitter, self).__init__()

class RandomSplitter(Splitter):
    """
    Random splitter.
    """
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset

class IndexSplitter(Splitter):
    """
    Split daatasets that has already been orderd. The first `frac_train` proportion
    is used for train set, the next `frac_valid` for valid set and the final `frac_test` 
    for test set.
    """
    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset

class ScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    
    Split dataset by Bemis-Murcko scaffolds
    """
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the 
                scaffold.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        
        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i in range(N):
            smiles = dataset[i][0]
            scaffold = generate_scaffold(dataset[i][0], include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)
        

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        #train_dataset = list(np.array(dataset)[train_idx])
        #valid_dataset = list(np.array(dataset)[valid_idx])
        #test_dataset = list(np.array(dataset)[test_idx])

        train_dataset = [dataset[i] for i in train_idx]
        valid_dataset = [dataset[i] for i in valid_idx]
        test_dataset = [dataset[i] for i in test_idx]

        return train_dataset, valid_dataset, test_dataset

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

class RandomScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    
    Split dataset by Bemis-Murcko scaffolds
    """
    def __init__(self):
        super(RandomScaffoldSplitter, self).__init__()
    
    def split(self, 
            dataset, 
            frac_train=None, 
            frac_valid=None, 
            frac_test=None,
            seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the 
                scaffold.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset[ind]['smiles'], include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
    


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        

        graph = self.graphs[index].to(self.device)
        
        return label, graph
    
def get_label():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['label']
#tox21,12     
def get_tox():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

#clintox,2
def get_clintox():
    
    return ['FDA_APPROVED', 'CT_TOX']

#sider,27
def get_sider():

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

#muv
def get_muv():
    
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',	'MUV-858','MUV-859']

def collate_fn(batch):
    labels, graphs = zip(*batch) 

    labels = torch.stack(labels)

    batched_graph = dgl.batch(graphs)

    return labels, batched_graph

def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
                return True
    return False

def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)

def creat_data(datafile,batch_size,train_ratio,vali_ratio,test_ratio):
    

    datasets = datafile

    directory_path = ''
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name):

        return True
    
    else:

        df = pd.read_csv('data/' + datasets + '.csv')
        if datasets == 'tox21':
            smiles_list, labels = df['smiles'], df[get_tox()] 
            #labels = labels.replace(0, -1)
            labels = labels.fillna(0)

        if datasets == 'muv':
            smiles_list, labels = df['smiles'], df[get_muv()]  
            labels = labels.fillna(0)

        if datasets == 'sider':
            smiles_list, labels = df['smiles'], df[get_sider()]  

        if datasets == 'clintox':
            smiles_list, labels = df['smiles'], df[get_clintox()] 
    

        if datasets in ['hiv','bbbp','bace']:
            smiles_list, labels = df['smiles'], df[get_label()] 
            
        #labels = labels.replace(0, -1)
        #labels = labels.fillna(0)

        #smiles_list, labels = df['smiles'], df['label']        
        #labels = labels.replace(0, -1)
        
        #labels, min_val, max_val = min_max_normalize(labels)
        data_list = []
        for i in range(len(smiles_list)):
            if i % 10000 == 0:
                print(i)

            smiles = smiles_list[i]
            
            #if has_isolated_hydrogens(smiles) == False and conformers_is_zero(smiles) == True :

            Graph_list = path_complex_mol(smiles)
            if Graph_list == False:
                continue

            else:
                if has_node_with_zero_in_degree(Graph_list):
                    continue
                
                else:
                    data_list.append([smiles, torch.tensor(labels.iloc[i].to_numpy()),Graph_list])



        #data_list = [['occr',albel,[c_size, features, edge_indexs],[g,liearn_g]],[],...,[]]

        print('Graph list was done!')

        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
        
        print('splitter was done!')
        

        
        train_label = []
        train_graph_list = []
        for tmp_train_graph in splitter[0]:
            
            train_label.append(tmp_train_graph[1])
            train_graph_list.append(tmp_train_graph[2])


        valid_label = []
        valid_graph_list = []
        for tmp_valid_graph in splitter[1]:
            valid_label.append(tmp_valid_graph[1])
            
            valid_graph_list.append(tmp_valid_graph[2])

        test_label = []
        test_graph_list = []
        for tmp_test_graph in splitter[2]:
            test_label.append(tmp_test_graph[1])
            test_graph_list.append(tmp_test_graph[2])

        #batch_size = 256

        torch.save({
            'train_label': train_label,
            'train_graph_list': train_graph_list,
            'valid_label': valid_label,
            'valid_graph_list': valid_graph_list,
            'test_label': test_label,
            'test_graph_list': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,  
        }, datafile +'.pth')






#################################### DATA PROCESSING 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('mps')
    print('The code uses MPS!!!')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

seed = 23
set_seed(seed)

batch_size = 128
train_ratio = 0.8
vali_ratio = 0.1
test_ratio = 0.1
target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
data_vec = ['bace','bbbp','hiv','clintox','sider','muv','tox21']

print('Now create dataset for - bace')
creat_data('bace', batch_size, train_ratio, vali_ratio, test_ratio)
'''
for dataset in data_vec:
    print('Now create dataset for ',dataset)
    creat_data(dataset, batch_size, train_ratio, vali_ratio, test_ratio)
'''
