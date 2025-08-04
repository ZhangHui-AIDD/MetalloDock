#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import copy
import MDAnalysis as mda
import networkx as nx
import numpy as np
import rmsd
from rdkit import Chem, RDLogger
from rdkit import Geometry
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.optimize import differential_evolution
from uff_build import uff_geomopt   
from minimize_utils import minimizing_pipeline
import time
import os
from Bio import PDB
from functools import partial
print = partial(print, flush=True)

# part of this code taken from EquiBind https://github.com/HannesStark/EquiBind
RDLogger.DisableLog('rdApp.*')
RMSD = AllChem.AlignMol


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)

def get_torsion_bonds(mol):
    torsions_list = []
    G = nx.Graph()
    # for i, atom in enumerate(mol.GetAtoms()):
    #     G.add_node(i)
    # nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list


# GeoMol
def get_torsions(mol_list):
    # print('USING GEOMOL GET TORSIONS FUNCTION')
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def torsional_align(rdkit_mol, pred_conf, rotable_bonds):
    for rotable_bond in rotable_bonds:
        diheral_angle = GetDihedral(pred_conf, rotable_bond)
        SetDihedral(rdkit_mol.GetConformer(0), rotable_bond, diheral_angle)
    return rdkit_mol

def random_torsion(mol):
    rotable_bonds = get_torsions([mol])
    torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=len(rotable_bonds))
    for idx, rotable_bond in enumerate(rotable_bonds):
        SetDihedral(mol.GetConformer(0), rotable_bond, torsion_updates[idx])
    return mol

def mmff_func(mol):
    mol_mmff = copy.deepcopy(mol)
    AllChem.MMFFOptimizeMoleculeConfs(mol_mmff, mmffVariant='MMFF94s')
    for i in range(mol.GetNumConformers()):
        coords = mol_mmff.GetConformers()[i].GetPositions()
        for j in range(coords.shape[0]):
            mol.GetConformer(i).SetAtomPosition(j,
                                                Geometry.Point3D(*coords[j]))

def init_mol_pos(mol):
    feed_back = [1, 1]
    while feed_back[0] == 0:
        feed_back = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
    return mol


def set_rdkit_mol_position(rdkit_mol, position):
    for j in range(position.shape[0]):
        rdkit_mol.GetConformer().SetAtomPosition(j,
                                            Geometry.Point3D(*position[j]))
    return rdkit_mol

def position_align_mol(rdkit_mol, refer_mol):
    rmsd = AllChem.AlignMol(rdkit_mol, refer_mol)
    return rmsd

def position_align_np(rdkit_mol, refer_mol, algo='kabsch'):
    A = rdkit_mol.GetConformer().GetPositions()
    B = refer_mol.GetConformer().GetPositions()
    B_center = rmsd.centroid(B)
    A -= rmsd.centroid(A)
    B -= B_center
    rmsd.quaternion_rotate
    if algo == 'kabsch':
        U = rmsd.kabsch(A, B)
    else: # quaternion
        U = rmsd.quaternion_rotate(A, B)
    A = np.dot(A, U)
    A += B_center
    set_rdkit_mol_position(rdkit_mol=rdkit_mol, position=A)



def out_pose(data, out_dir, out_init=False):
    for idx, mol in enumerate(data['ligand'].mol):
        if out_init:
            # random position
            pos_init = data['ligand'].pos[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
            random_mol = copy.deepcopy(mol)
            random_mol = set_rdkit_mol_position(random_mol, pos_init)
            random_file = f'{out_dir}/{data.pdb_id[idx]}_init_pos.sdf'
            Chem.MolToMolFile(random_mol, random_file)
        # correct pos
        pos_pred = data.pos_preds[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
        pred_mol = set_rdkit_mol_position(copy.deepcopy(mol), pos_pred)
        pred_file = f'{out_dir}/{data.pdb_id[idx]}_pred.sdf'
        Chem.MolToMolFile(pred_mol, pred_file)
       


def correct_one(mol, pos_pred, method='ff'):
    # set pos
    correct_mol = copy.deepcopy(mol)
    raw_mol = copy.deepcopy(mol)
    raw_mol = set_rdkit_mol_position(raw_mol, pos_pred)
    # FF
    if method == 'ff':
        correct_mol = set_rdkit_mol_position(correct_mol, pos_pred)
        try:
            AllChem.MMFFOptimizeMolecule(correct_mol, maxIters=10)
        except:
            print('FF optimization failed')
    else:
        # get torsion_bonds
        rotable_bonds = get_torsions([raw_mol])
        # torsional align
        correct_mol = torsional_align(rdkit_mol=correct_mol, pred_conf=raw_mol.GetConformer(), rotable_bonds=rotable_bonds)
        position_align_np(rdkit_mol=correct_mol, refer_mol=raw_mol)
    return correct_mol, raw_mol


def delete_mg_bond(protein_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(protein_path.replace('.pdb','_uff.pdb'))
    return protein_path.replace('.pdb','_uff.pdb')

def correct_one_uff(mol, pos_pred, protein_dir):
    if os.path.isfile(protein_dir):
        # set pos
        rd_mol = copy.deepcopy(mol)
        rd_mol = set_rdkit_mol_position(rd_mol, pos_pred)
        protein_file = protein_dir
        pkt_mol = Chem.MolFromPDBFile(protein_file)
        try:
            opt_mol = uff_geomopt(rd_mol,pkt_mol)
        except:
            uff_protein_file = delete_mg_bond(protein_file)
            pkt_mol = Chem.MolFromPDBFile(uff_protein_file)
            opt_mol = uff_geomopt(rd_mol,pkt_mol)
    else:
        # set pos
        rd_mol = copy.deepcopy(mol)
        rd_mol = set_rdkit_mol_position(rd_mol, pos_pred)
        protein_file = os.path.join(protein_dir,protein_dir.split('/')[-1]+'_pro_pocket.pdb') 
        pkt_mol = Chem.MolFromPDBFile(protein_file)
        try:
            opt_mol = uff_geomopt(rd_mol,pkt_mol)
        except:
            uff_protein_file = delete_mg_bond(protein_file)
            pkt_mol = Chem.MolFromPDBFile(uff_protein_file)
            opt_mol = uff_geomopt(rd_mol,pkt_mol)
    return opt_mol

def correct_one_energy_min(protein_dir, ligand_mol, pdbid, out_dir):
    if os.path.isfile(protein_dir):
        protein_file = protein_dir
        try:
            energy_min_corrected_mol = minimizing_pipeline(protein_file, ligand_mol, pdbid, out_dir)
        except:
            pass
    else:
        protein_file = os.path.join(protein_dir,pdbid,pdbid+'_pro_pocket.pdb') 
        try:
            energy_min_corrected_mol = minimizing_pipeline(protein_file, ligand_mol, pdbid, out_dir)
        except:
            pass
    return energy_min_corrected_mol



def correct_pos(data, out_dir, protein_path, min_method, out_corrected=True):
    poses = []
    for idx, mol in enumerate(data['ligand'].mol):
        if out_corrected:
            # correct pos without pocket
            pos_pred = data.pos_preds[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) 
            pos_true = data['ligand'].xyz[data['ligand'].batch==idx].cpu().numpy().astype(np.float64) 
            ff_corrected_mol, uncorrected_mol = correct_one(mol, pos_pred, method='ff')
            align_corrected_mol, uncorrected_mol = correct_one(mol, pos_pred, method='align')
            ff_corrected_file = f'{out_dir}/{data.pdb_id[idx]}_ff.sdf'
            try:
                Chem.MolToMolFile(ff_corrected_mol, ff_corrected_file)
            except:
                print(f'save {ff_corrected_file} failed')
                pass
            align_corrected_file = f'{out_dir}/{data.pdb_id[idx]}_align.sdf'
            try:
                Chem.MolToMolFile(align_corrected_mol, align_corrected_file)
            except:
                print(f'save {ff_corrected_file} failed')
                pass
            # for energy_minimizing (using pocket)
            if min_method == 'uff':
                energy_min_corrected_mol = correct_one_uff(mol, pos_pred, protein_path)
                uff_corrected_file = f'{out_dir}/{data.pdb_id[idx]}_uff.sdf'
                try:
                    Chem.MolToMolFile(energy_min_corrected_mol, uff_corrected_file)
                except:
                    print(f'save {uff_corrected_file} failed')
                    pass
            else:
                random_mol = copy.deepcopy(mol)
                pred_mol = set_rdkit_mol_position(random_mol, pos_pred)
                energy_min_corrected_mol = correct_one_energy_min(protein_path, pred_mol, data.pdb_id[idx], out_dir)
            poses.append([align_corrected_mol.GetConformer().GetPositions(), ff_corrected_mol.GetConformer().GetPositions(), energy_min_corrected_mol.GetConformer().GetPositions()])
    return poses

if __name__ == '__main__':
    mol = correct_uff(Chem.MolFromMolFile('/home/huizhang/code/KarmaDockGPT/utils/8a7r/8a7r_pred_uncorrected.sdf'), '/home/huizhang/dataset/equibind_metallo_test/8a7r/8a7r_pro_pocket.pdb', '/home/huizhang/code/KarmaDockGPT/utils/8a7r/8a7r_uff.sdf')
    Chem.MolToMolFile(mol, '/home/huizhang/code/KarmaDockGPT/utils/8a7r/8a7r_uff.sdf')