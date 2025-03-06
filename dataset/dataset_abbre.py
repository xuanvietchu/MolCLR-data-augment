import os
import csv
import re
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType as BT
from rdkit.Chem import AllChem
from abbrs import ABBREVIATIONS, ABBREVIATIONS_VOCAB, ABBREVIATION_PATTERNS
from rdkit.Chem import Draw
from visualize import visualize_molecule
from rdkit import Chem


ATOM_DICT = {atom: idx for idx, atom in enumerate(range(1, 193))}

CHIRALITY_DICT = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3
}

BOND_DICT = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1, 
    BT.TRIPLE: 2,
    BT.AROMATIC: 3
}
BONDDIR_DICT = {
    Chem.rdchem.BondDir.NONE: 0, 
    Chem.rdchem.BondDir.ENDUPRIGHT: 1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2
}

def apply_functional_group_abbreviations(mol):
    """Efficiently replace functional groups in a molecule with abbreviation tokens while preserving order."""
    patterns = list(ABBREVIATION_PATTERNS.items())
    random.shuffle(patterns)  # Randomize processing order

    for abbr, pattern in patterns:
        sub = ABBREVIATIONS[abbr]
        iso = ABBREVIATIONS_VOCAB[abbr]
        if not sub.probability:
            continue

        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            continue

        dummy = Chem.MolFromSmiles(f"[{iso}*]")
        atom = dummy.GetAtomWithIdx(0)
        atom.SetProp("abbr", abbr)

        placeholder = Chem.MolFromSmiles("[119*]")
        
        for match_atoms in matches:
            old_smiles = Chem.MolToSmiles(mol)

            if random.random() < sub.probability:
                mol = Chem.ReplaceSubstructs(mol, pattern, dummy, replaceAll=False)[0]
                img = visualize_molecule(mol, f"molecule with {sub.smarts} replaced by {abbr}")
                print(f"Replaced {sub.smarts} from {old_smiles} to {Chem.MolToSmiles(mol)} with {abbr}")
            else:
                mol = Chem.ReplaceSubstructs(mol, pattern, placeholder, replaceAll=False)[0]
                print(f"Skipped replacing {sub.smarts} with {abbr}")

            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace(".", ""))

        mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("[119*]"), pattern, replaceAll=True)[0]
        
    
    img = visualize_molecule(mol, f"molecule with abbreviations {Chem.MolToSmiles(mol)}")
    
    return mol


class MoleculeDataset(Dataset):
    def __init__(self, data_path, remove_header=False):
        super(Dataset, self).__init__()
        self.smiles_data = self.read_smiles(data_path, remove_header)

    def read_smiles(self, data_path, remove_header=False):
        smiles_data = []
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            if remove_header:
                next(csv_reader)
            for row in csv_reader:
                smiles_data.append(row[-1])
        return smiles_data

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        mol = apply_functional_group_abbreviations(mol)

        mol = Chem.AddHs(mol)

        
        N = mol.GetNumAtoms()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            if atom.HasProp("abbr"):  # Check if the atom has an abbreviation
                type_idx.append(vocab[atom.GetProp("abbr")])  # Use abbreviation index
                chirality_idx.append(0) # unspecified chirality for functional groups
                
            else:
                type_idx.append(ATOM_DICT[atom.GetAtomicNum()])  # Use atomic number if no abbreviation
                chirality_idx.append(CHIRALITY_DICT[atom.GetChiralTag()])
        
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_DICT[bond.GetBondType()],
                BONDDIR_DICT[bond.GetBondDir()]
            ])
            edge_feat.append([
                BOND_DICT[bond.GetBondType()],
                BONDDIR_DICT[bond.GetBondDir()]
            ])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __len__(self):
        return len(self.smiles_data)

class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, remove_header=False):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.remove_header = remove_header

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path, remove_header=self.remove_header)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader


if __name__ == "__main__":
    # python dataset/dataset_abbre.py

    # data_path = 'data/pubchem-20k-sample.txt'
    # dataset = MoleculeDataset(data_path=data_path)
    # print(dataset.__getitem__(0))

    
    # mol = Chem.MolFromSmiles('CCC(C)CC(NN)C1CCC(C)O1') 
    # mol = Chem.MolFromSmiles('CCOc1ccccc1N1C(=O)C(=O)N(CC(=O)c2cc(C)c(C)cc2C)C1=O') #1
    mol = Chem.MolFromSmiles('Cc1ccnc(N2CCC(C)C2C(=O)[O-])c1[N+](=O)[O-]') #2
    # mol = Chem.MolFromSmiles('CCOC(=O)C(N=[N+]=[N-])C(c1ccccc1)[NH+]1CCOCC1') #3
    # mol = Chem.MolFromSmiles('Cc1cc(N2C(=O)C3CC=CC(C)C3C2=O)no1') #4

    # print(ABBREVIATIONS_VOCAB)
    
    img = visualize_molecule(mol, "original molecule")
    # save image
    img.save("original_molecule.png")

    start = time.time()
    mol = apply_functional_group_abbreviations(mol)
    # print("Mol:", Chem.MolToSmiles(mol))

    # mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("C=CC"), Chem.MolFromSmiles("[119*]"), replaceAll=False)[0]
    # # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace(".", ""))
    # print("Mol:", Chem.MolToSmiles(mol))
    # img = visualize_molecule(mol, "molecule with abbreviations")

    # mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles("[119*]"), Chem.MolFromSmiles("C=CC"), replaceAll=True)[0]
    # # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace(".", ""))
    # print("Mol:", Chem.MolToSmiles(mol))
    # # remove . from the smiles of molecule
    # img = visualize_molecule(mol, "molecule with undo")


    end = time.time()
    print("Time taken to apply functional group abbreviations:", end-start)

