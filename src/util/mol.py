from signal import raise_signal
import numpy as np

from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Geometry import Point3D
from rdkit import RDLogger

from copy import deepcopy

import warnings

from .constants import BOND_LEN
# 关闭冗余日志和警告
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def gen_3d(smi: str, n_conf: int = 100) -> List:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        if bool:
            print("Error[iaw]>: can not read mol from `{}`".format(smi))
        return []
    mol = Chem.AddHs(mol)  
    AllChem.EmbedMultipleConfs(mol, numConfs = n_conf, useRandomCoords=True)
    opt_mol = []
    for conf_id in range(n_conf):
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()

        new_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
        opt_mol.append(new_mol)
    return opt_mol


def init_sub(fp: str, n_conf:int = 100, debug: bool = False) -> List[List[Chem.Mol]]:
    sub_mol = []
    with open(fp, "r", encoding="utf-8") as f:
        subs1 = [line.strip() for line in f if line.strip()]
    for smi in subs1:
        opt_mol = gen_3d(smi = smi, n_conf = n_conf)
        if opt_mol:
            sub_mol.append(opt_mol)
    return sub_mol

def sub_dummy(mol: Chem.Mol) -> Tuple[int, int]:
    """
        (-1, -1) error
        (dummy_idx, neighbor_idx) correct
    """
    assert len([1 for i_atm in mol.GetAtoms() if i_atm.GetSymbol() == '*']) == 1, print("Error[iaw]>: Substituent must contain exactly one dummy atom ('*')!")
    
    for i_atm in mol.GetAtoms():
        if i_atm.GetSymbol() == '*':
            _atm_nhb = i_atm.GetNeighbors()
            assert len(_atm_nhb) == 1, print("Error[iaw]>: Dummy atom has more than one neighbor!")
            _atm_nhb = _atm_nhb[0]
            return (i_atm.GetIdx(), _atm_nhb.GetIdx())
    
    return (-1, -1)

def scaffold_dummy(mol:Chem.Mol, h_idx:int) -> Tuple[int, int]:
    """
        (-1, -1) error
        (h_idx, neighbor_idx) correct
    """
    h_atom = mol.GetAtomWithIdx(h_idx)
    assert h_atom.GetSymbol() == 'H', print("Error[iaw]>: The specified index does not correspond to a hydrogen atom!")
    
    nhb_atoms = h_atom.GetNeighbors()
    assert len(nhb_atoms) == 1, print("Error[iaw]>: The hydrogen atom has more than one neighbor!")
    nhb_atom = nhb_atoms[0]
    
    return (h_idx, nhb_atom.GetIdx())

def align_sub(scaffold_mol: Chem.Mol, sub_mol: Chem.Mol, scaffold_idx: Tuple[int, int], sub_idx: Tuple[int, int]) -> Chem.Mol:

    def Rotation(axis, theta):
        """返回绕 axis 转 theta 的 3×3 旋转矩阵"""
        axis = axis / np.linalg.norm(axis)
        ux, uy, uz = axis
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([
            [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s,  ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
        ])
        return rot

    dummy_scaffold_idx, neighbor_scaffold_idx = scaffold_idx
    dummy_sub_idx, neighbor_sub_idx = sub_idx

    scaffold_conf = scaffold_mol.GetConformer()
    sub_conf = sub_mol.GetConformer()
    scaffold_dummy_pos = scaffold_conf.GetAtomPosition(dummy_scaffold_idx)
    scaffold_neighbor_pos = scaffold_conf.GetAtomPosition(neighbor_scaffold_idx)
    sub_dummy_pos = sub_conf.GetAtomPosition(dummy_sub_idx)
    sub_neighbor_pos = sub_conf.GetAtomPosition(neighbor_sub_idx)

    scaffold_vector = scaffold_dummy_pos - scaffold_neighbor_pos
    scaffold_vector /= np.linalg.norm(scaffold_vector)

    sub_vector = sub_neighbor_pos - sub_dummy_pos
    sub_vector /= np.linalg.norm(sub_vector)

    cross = np.cross(sub_vector, scaffold_vector)
    dot = np.dot(sub_vector, scaffold_vector)

    if abs(dot + 1.0) < 1e-6:            # 反向 180°
        axis = np.array([1, 0, 0]) if abs(sub_vector[0]) < 0.9 else np.array([0, 1, 0])
        cross = np.cross(sub_vector, axis)
        cross /= np.linalg.norm(cross)
        rot_mat = Rotation(cross, np.pi)
    elif abs(dot - 1.0) < 1e-6:          # 已同向
        rot_mat = np.eye(3)
    else:
        angle = np.arccos(dot)
        rot_mat = Rotation(cross, angle)

    aligned = Chem.Mol(sub_mol)            
    aligned.RemoveAllConformers() 
    new_conf = Chem.Conformer(sub_mol.GetNumAtoms())
    for i in range(sub_mol.GetNumAtoms()):
        pos = np.array(sub_conf.GetAtomPosition(i))
        pos_rot = rot_mat @ (pos - sub_dummy_pos)
        pos_new = pos_rot + scaffold_neighbor_pos
        new_conf.SetAtomPosition(i, Point3D(*pos_new))

    aligned.AddConformer(new_conf)
    aligned.UpdatePropertyCache(False)

    return aligned

def merge_mol(scaffold_mol: Chem.Mol, aligned_sub_mol: Chem.Mol, scaffold_idx: Tuple[int, int], sub_idx: Tuple[int, int]) -> Tuple[Chem.Mol, int]:
    dummy_scaffold_idx, neighbor_scaffold_idx = scaffold_idx
    dummy_sub_idx, neighbor_sub_idx = sub_idx
    
    neighbor_scaffold = scaffold_mol.GetAtomWithIdx(neighbor_scaffold_idx).GetSymbol().upper()
    neighbor_scaffold_pos = scaffold_mol.GetConformer().GetAtomPosition(neighbor_scaffold_idx)

    neighbor_sub = aligned_sub_mol.GetAtomWithIdx(neighbor_sub_idx).GetSymbol().upper()
    neighbor_sub_pos = aligned_sub_mol.GetConformer().GetAtomPosition(neighbor_sub_idx)

    mol1 = Chem.EditableMol(Chem.Mol(scaffold_mol))
    mol1.RemoveAtom(dummy_scaffold_idx)
    mol1 = mol1.GetMol() 

    
    mol2 = Chem.EditableMol(Chem.Mol(aligned_sub_mol))
    mol2.RemoveAtom(dummy_sub_idx)
    mol2 = mol2.GetMol() 

    # 获取键长
    if (neighbor_scaffold, neighbor_sub) in BOND_LEN.keys():
        bond_len = BOND_LEN[(neighbor_scaffold, neighbor_sub)]
    elif (neighbor_sub, neighbor_scaffold) in BOND_LEN.keys():
        bond_len = BOND_LEN[(neighbor_sub, neighbor_scaffold)]
    else:
        raise RuntimeError("Error[iaw]>: can not support bond between {} and {}.".format(neighbor_scaffold, neighbor_sub))


    direction_vector = neighbor_sub_pos - neighbor_scaffold_pos
    normalized_direction_vector = direction_vector / np.linalg.norm(neighbor_sub_pos - neighbor_scaffold_pos)
    translation_vector = normalized_direction_vector * (bond_len - np.linalg.norm(neighbor_sub_pos - neighbor_scaffold_pos))

    # 平移替换B
    mol2_conf = mol2.GetConformer()
    for i in range(mol2.GetNumAtoms()):
        pos = mol2_conf.GetAtomPosition(i)
        new_pos = pos + translation_vector
        mol2_conf.SetAtomPosition(i, new_pos)

    final = Chem.CombineMols(mol1, mol2)
    combo = Chem.RWMol(final)
    
    sub_start_idx = mol1.GetNumAtoms() 
    neighbor_scaffold_idx_new = neighbor_scaffold_idx
    neighbor_sub_idx_new = neighbor_sub_idx + sub_start_idx -1
    
    combo.AddBond(neighbor_scaffold_idx_new,
                  neighbor_sub_idx_new,
                  Chem.BondType.SINGLE)
    final = combo.GetMol() 

    final.UpdatePropertyCache(False)

    
    return final, sub_start_idx