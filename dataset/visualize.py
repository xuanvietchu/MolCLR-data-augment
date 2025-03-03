from rdkit.Chem import Draw

def visualize_molecule(mol, name="Molecule"):
    """
    Visualizes an RDKit molecule with annotations for atoms that have an "abbr" property.
    
    Parameters:
        mol (Chem.Mol): The RDKit molecule.
    
    Returns:
        PIL.Image.Image: An image of the molecule with highlighted dummy atoms and their abbreviation labels.
    """
    # Collect indices and labels for atoms annotated with "abbr"
    atom_labels = {}
    highlight_atoms = []
    for atom in mol.GetAtoms():
        if atom.HasProp("abbr"):
            idx = atom.GetIdx()
            highlight_atoms.append(idx)
            atom_labels[idx] = atom.GetProp("abbr")
    
    # Generate and return the image with highlighted atoms and annotation labels.
    img = Draw.MolToImage(
        mol,
        size=(1000, 1000),
        highlightAtoms=highlight_atoms,
        highlightAtomLabels=atom_labels,
        legend=name
    )
    img.show()

    return img


def visualize_smiles(mol, name="Molecule"):
    """
    Visualizes a SMILES string with annotations for atoms that have an "abbr" property.
    
    Parameters:
        mol (str): The SMILES string.
    
    Returns:
        PIL.Image.Image: An image of the molecule with highlighted dummy atoms and their abbreviation labels.
    """
    mol = Chem.MolFromSmiles(mol)
    return visualize_molecule(mol, name=name)