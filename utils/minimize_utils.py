import os
from rdkit import Chem
from minimize_build import GetfixedPDB,GetFFGenerator,UpdatePose,GetPlatformPara,GetPlatform,Molecule,trySystem,read_molecule
import sys
from openmm.app import Modeller
from joblib import Parallel,delayed
import argparse
from tqdm import tqdm
from glob import glob
import warnings
import traceback
import time
import pandas as pd
import numpy as np
import copy
from Bio import PDB
import scipy.spatial as spatial
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import logging
for module in [
    "openff.toolkit",
    "openff.toolkit.typing.engines",
    "openff.toolkit.typing.engines.smirnoff.parameters",
    "openmmforcefields",
]:
    logging.getLogger(module).setLevel(logging.WARNING)

"""
"Some scripts is based on https://github.com/CAODH/SurfDock/blob/master/force_optimize/post_energy_minimize.py"
"""

def delete_mg_bond(protein_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(protein_path.replace('.pdb','_uff.pdb'))
    return protein_path.replace('.pdb','_uff.pdb')

def correct_uff(mol, protein_dir, output_file):
    rd_mol = copy.deepcopy(mol)
    protein_file = protein_dir
    pkt_mol = Chem.MolFromPDBFile(protein_file)
    try:
        opt_mol = uff_geomopt(rd_mol,pkt_mol)
    except:
        uff_protein_file = delete_mg_bond(protein_file)
        pkt_mol = Chem.MolFromPDBFile(uff_protein_file)
        opt_mol = uff_geomopt(rd_mol,pkt_mol)
    return opt_mol

def get_rd_atom_res_id(rd_atom):
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )

def get_pocket(lig_mol, rec_mol, max_dist=8):
    lig_coords = lig_mol.GetConformer().GetPositions()
    rec_coords = rec_mol.GetConformer().GetPositions()
    dist = spatial.distance.cdist(lig_coords, rec_coords)
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])
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
        converged = not uff.Minimize(maxIts=n_iters)
        n_tries -= 1
    conf = rd_mol_rw.GetConformer()
    for atom_idx in range(rd_mol.GetNumAtoms()):
        pos = uff_mol.GetConformer().GetAtomPosition(pkt_mol.GetNumAtoms() + atom_idx)
        conf.SetAtomPosition(atom_idx, pos)
    return rd_mol_rw


def minimizing_pipeline(protein_path, ligand_mol, pdb_id, out_dir, head_num=20, num_process=20, head_index=0, tail_index=-1, cuda=0):
    """
    Process protein-ligand docking files, minimize and save results.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.environ['OMP_NUM_THREADS'] = '1'
    """Init force field"""
    start_time = time.time()
    platform = GetPlatformPara()
    system_generator = GetFFGenerator(ignoreExternalBonds=True)
    system_generator_gaff = GetFFGenerator(small_molecule_forcefield='gaff-2.11', ignoreExternalBonds=True)

    receptor_path = protein_path
    fixer = GetfixedPDB(receptor_path)
    modeller = Modeller(fixer.topology, fixer.positions)
    protein_atoms = list(modeller.topology.atoms())
    try:
        dockingpose = ligand_mol
        lig_mol = Molecule.from_rdkit(dockingpose, allow_undefined_stereo=True)
        # set formal_charge using Gasteiger
        lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
        modeller = trySystem(system_generator_gaff, modeller, lig_mol, pdb_id)
        failed_create_system = False
    except Exception as e:
        # logger.error(f"ERROR in create system step! {e}")
        failed_create_system = True
        lig_mol = ligand_mol 
    if not failed_create_system:
        new_data = UpdatePose(ligand_mol, system_generator, modeller, protein_atoms, out_dir, pdb_id, receptor_path)
    else:
        new_data = 2   
    # selected the failed samples and try to use gaff-2.11 forcefield
    if new_data == 1:
        print(f'Minimized not completed: {pdb_id}. SDF will not be minimized by default forcefield, trying GAFF-2.11.')
        if not failed_create_system:
            new_data = UpdatePose(ligand_mol, system_generator_gaff, modeller, protein_atoms, out_dir, pdb_id, receptor_path)
        if new_data != 0:
            try:
                out_file = os.path.join(out_dir,pdb_id + '_minimized.sdf')
                lig_mol = correct_uff(ligand_mol, receptor_path, out_file)
            except Exception as e:
                logger.error(f"ERROR in UFF Minimization! {e}")
            return  lig_mol
    if new_data == 2:
        try:
            out_file = os.path.join(out_dir,pdb_id + '_minimized.sdf')   
            lig_mol = correct_uff(ligand_mol, receptor_path, out_file)
            writer = Chem.SDWriter(out_file)
            writer.write(lig_mol)
        except Exception as e:
            logger.error(f"ERROR in UFF Minimization! {e}")
        return  lig_mol
        
    end_time = time.time()
    # print(f"Time taken for optimizing molecules: {end_time - start_time:.2f} seconds")
    return  read_molecule(os.path.join(out_dir, pdb_id + '_minimized.sdf'), sanitize=False, calc_charges=False, remove_hs=True)


if __name__ == '__main__':
    pass
