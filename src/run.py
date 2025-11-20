import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .util.mol import *
from .util.opt import *

from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
import warnings
# 关闭冗余日志和警告
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from rich.progress import track
from rich.status import Status
from rich import print as rp 

import argparse

def process_sub(scaffold_mol: Chem.Mol, sub_mol: List, scaffold_idx: Tuple[int, int], min_dist: float = 1.2) -> Chem.Mol:
    out = {}
    min_collision_mol = None
    for i, sub in enumerate(sub_mol):
        dummy_sub_idx, neighbor_sub_idx =  sub_dummy(sub)
        align_sub_mol = align_sub( scaffold_mol = scaffold_mol
                                 , sub_mol = sub
                                 , scaffold_idx = scaffold_idx
                                 , sub_idx = (dummy_sub_idx,neighbor_sub_idx))

        # Chem.MolToPDBFile(align_sub_mol, "aligned_sub.pdb")
        final_mol, new_sub_idx_start  = merge_mol( scaffold_mol = scaffold_mol
                          , aligned_sub_mol = align_sub_mol
                          , scaffold_idx = scaffold_idx
                          , sub_idx = (dummy_sub_idx,neighbor_sub_idx))

        collision_min = check_collision(final_mol, new_sub_idx_start, min_dist)

        if collision_min == -1:
            min_collision_mol = deepcopy(final_mol)
            break
        else:
            out[str(i)] = (collision_min, deepcopy(final_mol))
    if not min_collision_mol:  
        sorted_out = sorted(out.items(), key=lambda item: item[1][0], reverse=True)
        min_collision_mol = sorted_out[0][1][1]
        #Chem.MolToPDBFile(min_collision_mol, "final.pdb")
    return min_collision_mol
    
def Parm():

    parser = argparse.ArgumentParser(description='XX')
    parser.add_argument('-scaffold'
                        , type = str
                        , nargs = 1
                        , help = '母核的PDB文件')
    parser.add_argument('-dummy'
                        , type = int
                        , nargs = 1
                        , help = '取代位点的H原子序号')
    parser.add_argument('-sub'
                        , type = str
                        , nargs = 1
                        , help = '取代基团TXT文件')
    parser.add_argument('-conf'
                        , type = int
                        , nargs=1
                        , help='sub构象数目 (100)'
                        , default=[100])
    parser.add_argument("-dist"
                        , type = float
                        , nargs=1
                        , help = '原子冲突阈值 (1.2)'
                        , default=[1.2])
    parser.add_argument('-out'
                        , type = str
                        , nargs=1
                        , help='输出文件路径')
    
    return parser.parse_args()


def main():
    parm = Parm()

    scaffold_path = parm.scaffold[0]
    scaffold_h_idx = parm.dummy[0]
    subs_file = parm.sub[0]
    n_conf = parm.conf[0]
    min_dist = parm.dist[0]
    out_path = parm.out[0]

    scaffold_mol = Chem.MolFromPDBFile(scaffold_path, removeHs=False)

    with Status("[cyan]Processing Gen Sun Conf...") as status:
        sub_mols = init_sub(subs_file, n_conf, debug=False)

    dummy_scaffold_idx, neighbor_scaffold_idx =  scaffold_dummy(scaffold_mol, scaffold_h_idx)
    for i, sub in track(enumerate(sub_mols)):
        #rp(sub)
        final_mol = process_sub(scaffold_mol, sub, (dummy_scaffold_idx, neighbor_scaffold_idx), min_dist)
        Chem.MolToPDBFile(final_mol, "{}/final-{}.pdb".format(out_path, i))


if __name__ == "__main__":
    main()   

