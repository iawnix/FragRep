from tkinter import NO
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
import numpy as np

import warnings



# 关闭冗余日志和警告
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def opt_sub_mol_mmff94(mol: Chem.Mol, sub_start_idx: int) -> Chem.Mol:
    new = Chem.Mol(mol)
    Chem.GetSSSR(new)        
    new.UpdatePropertyCache(False)

    props = AllChem.MMFFGetMoleculeProperties(new)
    ff = AllChem.MMFFGetMoleculeForceField(new, props)
    for i_atm in new.GetAtoms():
        idx = i_atm.GetIdx()
        if idx < sub_start_idx:
            ff.AddFixedPoint(idx) 
    ff.Minimize(maxIts=500)

    return new

def check_collision(mol: Chem.Mol, start_idx: int, min_dist: float = 1.2) -> float:
    conf = mol.GetConformer()
    coords1 = []
    coords2 = []

    for i in range(mol.GetNumAtoms()):
        if i < start_idx:
            coords1.append(np.array(conf.GetAtomPosition(i)))
        else:
            coords2.append(np.array(conf.GetAtomPosition(i)))
    coords1 = np.array(coords1)      
    coords2 = np.array(coords2)
    dist = np.linalg.norm(coords2[:, np.newaxis] - coords1,  axis=2)

    if np.any(dist < min_dist):
        min_dist = np.min(dist[dist <= min_dist])
    else:
        min_dist =  -1
    
    return min_dist
