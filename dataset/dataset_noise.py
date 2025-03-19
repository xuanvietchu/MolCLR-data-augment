import csv
import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import Data, Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def add_noise_to_node_features(x, noise_level=0.1):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise 


def add_noise_to_edge_features(edge_attr, noise_level=0.1):
    noise = np.random.normal(0, noise_level, edge_attr.shape)
    return edge_attr + noise


def read_smiles(data_path, remove_header=False):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        if remove_header:
            next(csv_reader)
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self, data_path, remove_header=False):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path, remove_header)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        # get atom features for mol
        type_idx, chirality_idx, atomic_number = [], [], []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
        x = torch.cat(
            [
                torch.tensor(type_idx, dtype=torch.long).view(-1, 1),
                torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1),
            ],
            dim=-1,
        )
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
            edge_feat.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ]
            )
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # add noise to node features
        x_i = add_noise_to_node_features(x, noise_level=0.1)
        x_j = add_noise_to_node_features(x, noise_level=0.1)

        # add noise to edge features
        edge_attr_i = add_noise_to_edge_features(edge_attr, noise_level=0.1)
        edge_attr_j = add_noise_to_edge_features(edge_attr, noise_level=0.1)

        # get data
        data_i = Data(x=x_i, edge_index=edge_index, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index, edge_attr=edge_attr_j)

        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(
        self, batch_size, num_workers, valid_size, data_path, remove_header=False
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.remove_header = remove_header

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(
            data_path=self.data_path, remove_header=self.remove_header
        )
        train_loader, valid_loader = self.get_train_validation_data_loaders(
            train_dataset
        )
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return train_loader, valid_loader


if __name__ == "__main__":
    """
    Tests
    """
    mol = Chem.MolFromSmiles("CCC")
    print(Chem.MolToSmiles(mol))

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()
    print(N, M)

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())
    print(type_idx)
    print(chirality_idx)
    print(atomic_number)

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)
    print(x1)
    print(x2)
    print(x)
    print(add_noise_to_node_features(x))

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append(
            [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
        )
        edge_feat.append(
            [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
        )
    print(row)
    print(col)
    print(edge_feat)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    print(edge_index)
    print(edge_attr)
    print(add_noise_to_edge_features(edge_attr))