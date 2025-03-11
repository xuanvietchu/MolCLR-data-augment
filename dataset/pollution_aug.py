from rdkit import Chem
import random


def add_random_atoms(mol, atom_symbol):
    """
    Add a random atom to the molecule and connect it to a random existing atom
    """
    new_mol = Chem.RWMol(mol)
    atom_idx = new_mol.AddAtom(Chem.Atom(atom_symbol))

    existing_atom_idx = random.choice(range(mol.GetNumAtoms()))
    if new_mol.GetBondBetweenAtoms(atom_idx, existing_atom_idx) is None:
        new_mol.AddBond(atom_idx, existing_atom_idx, Chem.BondType.SINGLE)

    return new_mol.GetMol()


if __name__ == "__main__":
    """
    Tests
    """
    mol = Chem.MolFromSmiles("CCO")
    mol = add_random_atoms(mol, "F")
    print(Chem.MolToSmiles(mol))
