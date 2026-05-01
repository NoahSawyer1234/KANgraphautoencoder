#####   Importing  ######
#   - PyTorch for network architecture
#   - DGL for graph network data structuring
#   - Jarvis for chemical attribute production
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from torch.utils.data import Dataset
import os
import pandas as pd
from rdkit.Chem import rdchem
from rdkit.Chem import Lipinski
from torch_geometric.data import Data

####    Function for bond (edge) length  #####
def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis

hybrid_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]

chiral_types = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
    ]

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) +1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

# Node features - 23 vars
def node_feature_vec(atom):
    atom_num = atom.GetAtomicNum()
    num_bonds = atom.GetDegree()
    hybridisation = atom.GetHybridization() #Needs to be processed from rdkit type
    hybrid_one_hot = one_hot_encoding(hybridisation,hybrid_types) #pushed to 6 vars 
    atom_charge = atom.GetFormalCharge()
    imp_valence = atom.GetValence(which=Chem.rdchem.ValenceType.IMPLICIT)
    num_hydrogens = atom.GetNumImplicitHs()
    in_ring = int(atom.IsInRing())
    is_aromatic = int(atom.GetIsAromatic())
    chirality = atom.GetChiralTag() #Needs to be processed from rdkit type
    chiral_one_hot = one_hot_encoding(chirality,chiral_types) #pushed to 3 vars
    mass = atom.GetMass()
    num_radicals = atom.GetNumRadicalElectrons()
    exp_valence = atom.GetValence(which=Chem.rdchem.ValenceType.EXPLICIT)
    van_der_waals_rad = Chem.GetPeriodicTable().GetRvdw(atom_num)
    covalent_rad = Chem.GetPeriodicTable().GetRcovalent(atom_num)
    is_acceptor = int(atom.HasProp('_HAcceptor') )
    is_donor = int(atom.HasProp('_HDonor'))

    return [atom_num,num_bonds] + hybrid_one_hot + [atom_charge,imp_valence,
                                                    num_hydrogens,in_ring,is_aromatic] +chiral_one_hot +[mass,num_radicals,
                                                                                                         exp_valence,
                                                                                                         van_der_waals_rad,
                                                                                                         covalent_rad,is_acceptor,
                                                                                                         is_donor]

stereo_types = [
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ
    ]

# 10 features (require conformer for distance)
def bond_feature_vec(bond,conf):
    bond_type = bond.GetBondTypeAsDouble() #Gives bond type as numeric rather than needing onehot
    is_conjugated = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())
    stereo = bond.GetStereo()
    stereo_one_hot = one_hot_encoding(stereo,stereo_types) # pushed to 3 binary

    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    dist = AllChem.GetBondLength(conf, idx1, idx2)

    #add some geometric bits
    is_strained = bond.IsInRingSize(3) or bond.IsInRingSize(4)
    is_strained = int(is_strained)
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    is_linear = (atom1.GetHybridization() == Chem.rdchem.HybridizationType.SP or atom2.GetHybridization() == Chem.rdchem.HybridizationType.SP)
    is_linear = int(is_linear)
    
    return [bond_type,is_conjugated,in_ring] + stereo_one_hot + [dist,1/dist**2,is_strained,is_linear]
    
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
    mol = Chem.AddHs(mol)  #Add hydrogens, important for conformers
    node_features = []
    node_coor = []
    edge_features = []

    if mmff_force_field(mol) == True:
        num_conformers = mol.GetNumConformers()
        if num_conformers > 0:
            AllChem.ComputeGasteigerCharges(mol)
            conf = mol.GetConformer()
            for i, atom in enumerate(mol.GetAtoms()):
                feat = node_feature_vec(atom)

                node_features.append(feat)
                pos = mol.GetConformer().GetAtomPosition(i)
                node_coor.append([pos.x, pos.y, pos.z])

            src_list, dst_list = [], []
            for bond in mol.GetBonds():
                src = bond.GetBeginAtomIdx()
                dst = bond.GetEndAtomIdx()
                src_list.append(src)
                src_list.append(dst)
                dst_list.append(dst)
                dst_list.append(src)
                
                bond_feat = bond_feature_vec(bond,conf)

                edge_features.append(bond_feat)
                edge_features.append(bond_feat)
            
            coor_tensor = torch.tensor(node_coor, dtype=torch.float32)
            node_feats = torch.tensor(node_features,dtype=torch.float32)
            edge_feats = torch.tensor(edge_features, dtype=torch.float32)
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

            g = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, pos=coor_tensor)
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
class ScaffoldSplitter(object):
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
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

        train_dataset = [dataset[i] for i in train_idx]
        valid_dataset = [dataset[i] for i in valid_idx]
        test_dataset = [dataset[i] for i in test_idx]

        return train_dataset, valid_dataset, test_dataset

def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold 
    

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

        data_list = []
        for i in range(len(smiles_list)):
            #if i % 100 == 0:
                #print("Currently on molecule ",i)
            smiles = smiles_list[i]
            graph_list = path_complex_mol(smiles)
            if graph_list == False:
                continue
            else:
                label = torch.tensor(labels.iloc[i].to_numpy(), dtype=torch.float32)
                graph_list.y = label
                data_list.append([smiles, graph_list])
        #print('Graph list was done!')
        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
        #print('splitter was done!')
  
        train_graph_list = []
        for tmp_train_graph in splitter[0]:
            train_graph_list.append(tmp_train_graph[1])

        valid_graph_list = []
        for tmp_valid_graph in splitter[1]:
            valid_graph_list.append(tmp_valid_graph[1])

        test_graph_list = []
        for tmp_test_graph in splitter[2]:
            test_graph_list.append(tmp_test_graph[1])

        torch.save({
            'train': train_graph_list,
            'valid': valid_graph_list,
            'test': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,
        }, datafile + f'_{batch_size}.pth')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

#################################### DATA PROCESSING 
def graph_processing(data,batch,train,test,valid):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        #print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        #print('The code uses CPU!!!')

    seed = 23
    set_seed(seed)

    batch_size = batch
    train_ratio = train
    vali_ratio = valid
    test_ratio = test
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    data_vec = ['bace','bbbp','hiv','clintox','sider','muv','tox21']

    creat_data(data, batch_size, train_ratio, vali_ratio, test_ratio)
'''
for dataset in data_vec:
    print('Now create dataset for ',dataset)
    creat_data(dataset, batch_size, train_ratio, vali_ratio, test_ratio)
'''
