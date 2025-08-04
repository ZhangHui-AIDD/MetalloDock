import os
import os.path as osp
from rdkit import Chem 
from rdkit.Chem import AllChem,SDWriter
from rdkit import Chem
import scipy.spatial as spatial
import numpy as np
import time
from functools import partial
print = partial(print, flush=True)

def get_rd_atom_res_id(rd_atom):
    '''
    Return an object that uniquely
    identifies the residue that the
    atom belongs to in a given PDB.
    '''
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )

def get_pocket(lig_mol, rec_mol, max_dist=8):
    lig_coords = lig_mol.GetConformer().GetPositions()
    rec_coords = rec_mol.GetConformer().GetPositions()
    dist = spatial.distance.cdist(lig_coords, rec_coords)

    # indexes of atoms in rec_mol that are
    #   within max_dist of an atom in lig_mol
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])

    # determine pocket residues
    pocket_res_ids = set()
    for i in pocket_atom_idxs:
        atom = rec_mol.GetAtomWithIdx(int(i))
        res_id = get_rd_atom_res_id(atom)
        pocket_res_ids.add(res_id)

    pkt_mol = rec_mol
    pkt_mol = Chem.RWMol(pkt_mol)
    for atom in list(pkt_mol.GetAtoms()):
        res_id = get_rd_atom_res_id(atom)
        if res_id not in pocket_res_ids:
            pkt_mol.RemoveAtom(atom.GetIdx())

    Chem.SanitizeMol(pkt_mol)
    return pkt_mol

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

def write_sdf(mol_list,file):
    writer = Chem.SDWriter(file)
    for i in mol_list:
        writer.write(i)
    writer.close()
    

def uff_geomopt(rd_mol, pkt_mol, lig_constraint=None, n_iters=20, n_tries=2, lig_h=False, pkt_h=False):
    pkt_mol = get_pocket(rd_mol, pkt_mol, max_dist=6)   
    if lig_h:
        rd_mol = Chem.AddHs(rd_mol, addCoords=True)
    if pkt_h:
        pkt_mol = Chem.AddHs(pkt_mol, addCoords=True)
    rd_mol_rw = Chem.RWMol(rd_mol)
    uff_mol = Chem.CombineMols(pkt_mol, rd_mol_rw)
    Chem.SanitizeMol(uff_mol)
    uff = AllChem.UFFGetMoleculeForceField(uff_mol, confId=0, ignoreInterfragInteractions=False)
    uff.Initialize()
    for i in range(pkt_mol.GetNumAtoms()):
        uff.AddFixedPoint(i) 
    if lig_constraint is not None:
        for i in lig_constraint:
            atom_idx_in_uff = pkt_mol.GetNumAtoms() + i 
            uff.AddFixedPoint(atom_idx_in_uff)
    converged = False 
    while n_tries > 0 and not converged:
        # print('.', end='', flush=True)
        converged = not uff.Minimize(maxIts=n_iters)
        n_tries -= 1 
    print(flush=True)
    print("Performed UFF with binding site...")
    
    # Extract optimized coordinates from uff_mol and apply them to rd_mol_rw 
    conf = rd_mol_rw.GetConformer()
    for atom_idx in range(rd_mol.GetNumAtoms()):
        pos = uff_mol.GetConformer().GetAtomPosition(pkt_mol.GetNumAtoms() + atom_idx)
        conf.SetAtomPosition(atom_idx, pos)
    return rd_mol_rw


if __name__ == '__main__':
    pdbid= '7fur_ATP'
    pkt_mol = Chem.MolFromPDBFile(f'/home/huizhang/code/MetalloDock/utils/{pdbid}/{pdbid}_pro_pocket.pdb')
    rd_mol = read_sdf(f'/home/huizhang/code/MetalloDock/utils/7fur_ATP/7furATP_ligand.sdf')[0]
    start_time = time.perf_counter()
    opt_mol = uff_geomopt(rd_mol,pkt_mol)
