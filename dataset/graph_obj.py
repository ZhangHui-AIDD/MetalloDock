#!usr/bin/env python3
# -*- coding:utf-8 -*-
import copy
import glob
import prody
import os
from queue import Queue
from joblib import load
from random import random, choice, randint, choices
import sys
import MDAnalysis as mda
from functools import partial 
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import RandomTransform
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
RDLogger.DisableLog("rdApp.*")
# dir of current
from utils.fns import load_graph, save_graph
from dataset.metalloprotein_feature import get_metalloprotein_feature, get_metalloprotein_feature_vs
from dataset.ligand_feature import get_patom_feature, get_ligand_feature
from utils.post_processing import mmff_func
from pprint import pprint
print = partial(print, flush=True)
   

class PDBBindGraphDataset(Dataset):

    def __init__(self, src_dir, pdb_ids, dst_dir, voc_file, n_job=1, verbose=False):
        '''

        :param src_dir: path for saving pocket file and ligand file
        :param pdb_ids: pdb id of protein file
        :param dst_dir: path for saving graph file
        :param pocket_centers: the center of pocket (the center of the crystal ligand), (Num of complex, 3) np.array
        :param dataset_type: in ['train', 'valid', 'test']
        :param n_job: if n_job == 1: use for-loop;else: use multiprocessing
        :param on_the_fly: whether to get graph from a totoal graph list or a single graph file
        _______________________________________________________________________________________________________
        |  mode  |  generate single graph file  |  generate integrated graph file  |  load to memory at once  |
        |  False |          No                  |              Yes                 |            Yes           |
        |  True  |          Yes                 |              No                  |            No            |
        |  Fake  |          Yes                 |              No                  |            Yes           |
        _______________________________________________________________________________________________________
        '''
        self.src_dir = src_dir
        self.pdb_ids = pdb_ids
        self.dst_dir = dst_dir
        os.makedirs(dst_dir, exist_ok=True)
        self.n_job = n_job
        self.verbose = verbose
        self.graph_labels = []
        self.frag_voc = load(voc_file)
        self.pre_process()

    def pre_process(self):
        self._generate_graph_on_the_fly()


    def _generate_graph_on_the_fly(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly')
        single_process = partial(self._single_process, return_graph=False, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                single_process(idx)
        else:
            pool = Pool(self.n_job)
            pool.map(single_process, idxs)
            pool.close()
            pool.join()

    def _single_process(self, idx, return_graph=False, save_file=False):
        pdb_id = self.pdb_ids[idx]
        dst_file = f'{self.dst_dir}/{pdb_id}.pkl'
        if os.path.exists(dst_file):
            # reload graph
            if return_graph:
                return load_graph(dst_file)
        else:
            # generate graph
            src_path_local = f'{self.src_dir}/{pdb_id}'               
            pocket_pdb = f'{src_path_local}/{pdb_id}_pro_pocket.pdb'         
            complex_pocket_pdb = f'{src_path_local}/{pdb_id}_com_pocket.pdb' 
            ligand_crystal_pdb = f'{src_path_local}/{pdb_id}_ligand.pdb'  
            ligand_crystal_sdf = f'{src_path_local}/{pdb_id}_ligand.sdf'               
            try:
                data = get_graph_v1(pocket_pdb=pocket_pdb,
                                    complex_pocket_pdb=complex_pocket_pdb,
                                    ligand_crystal_pdb=ligand_crystal_pdb,
                                    ligand_crystal_mol2='',
                                    ligand_crystal_sdf=ligand_crystal_sdf)
                data.pdb_id = pdb_id
                if save_file:
                    save_graph(dst_file, data)
                if return_graph:
                    return data
            except:
                print(f'{pdb_id} error')
                return None

    def __getitem__(self, idx):
        data = self._single_process(idx=idx, return_graph=True, save_file=False)
        data['ligand'].seq_order, data['ligand'].seq_parent = torch.tensor([]), torch.tensor([])
        start_index = data['metal'].start_ligand_index.item()
        if start_index!= 9999:
            seq_order, seq_parent = BFS_search(data['ligand'].mol, start_atom_idx=start_index, random_seed=None)
            seq_parent[0] = data['metal'].start_metal_index
            # example 
            # Sorted atom indices: [0, 1, 2, 3, 4, 5, 37, 6, 8, 7, 9, 10, 11, 12, 17, 13, 18, 14, 19, 20, 15, 16, 21, 33, 22, 34, 23, 32, 35, 36, 24, 31, 25, 26, 27, 28, 29, 30]
            # Parent of each atom: [9999, 0, 1, 2, 3, 4, 4, 5, 37, 6, 8, 9, 9, 11, 11, 12, 17, 13, 18, 18, 14, 14, 20, 20, 21, 33, 22, 22, 34, 34, 23, 32, 24, 25, 26, 27, 27, 27]
            data['ligand'].seq_order = torch.tensor(seq_order[:], dtype=torch.long)
            data['ligand'].seq_parent = torch.tensor(seq_parent[:], dtype=torch.long)
        return data

    def __len__(self):
        return len(self.pdb_ids)

    
class MetalloVSTestGraphDataset_Fly(Dataset):
    def __init__(self, protein_file, ligand_path, voc_file):
        self.ligand_names = []
        self.ligand_smis = []
        self.protein_file = protein_file
        self.ligand_path = ligand_path
        torch.set_num_threads(1)
        self.voc_dict = voc_file
    
    def _get_mol(self, idx):
        return None
    
    def _single_process(self, idx):
        # torch.set_num_threads(1)
        ligand_name = self.ligand_names[idx]
        # generate graph
        cry_ligand_mol = self._get_mol(idx)
        data = get_graph_vs(pocket_pdb=self.protein_file,
                            complex_pocket_pdb=self.protein_file,
                            ligand_crystal_pdb='',
                            ligand_crystal_mol2='',
                            ligand_crystal_sdf=self.ligand_path)
        if data == None:
            pass
        data.pdb_id = ligand_name
        return data

    def __getitem__(self, idx):
        try:
            data = self._single_process(idx)
            if data == None:
                data['ligand'].pos = random_rotation(shuffle_center(data['ligand'].pos))  
        except:
            print(f'{self.ligand_names[idx]} error')
            return None
        return data

    def __len__(self):
        return len(self.ligand_names)
    

class VSTestGraphDataset_Fly_SDFMOL2_Refined(MetalloVSTestGraphDataset_Fly):
    '''refined the ligand conformation initialized with provied pose from SDF/MOL2 files'''
    def __init__(self, protein_file, ligand_path, ligand_files, voc_file, my_device):
        super().__init__(protein_file, ligand_path, voc_file)
        self.protein_file = protein_file
        self.ligand_names = ligand_files
        self.my_device = my_device

    def _get_mol(self, idx):
        ligand_name = self.ligand_names[idx]
        lig_file_sdf = f'{self.ligand_path}/{ligand_name}.sdf'
        lig_file_mol2 = f'{self.ligand_path}/{ligand_name}.mol2'
        mol = file2conformer(lig_file_sdf, lig_file_mol2)
        return mol, ligand_name
    
    def _get_sdf(self, idx):
        ligand_name = self.ligand_names[idx]
        lig_file_sdf = f'{self.ligand_path}/{ligand_name}.sdf'
        return lig_file_sdf, ligand_name

    def _single_process(self, idx):
        torch.set_num_threads(1)
        # generate graph
        try:
            ligand_file_sdf, ligand_name = self._get_sdf(idx)
            data = get_graph_vs(pocket_pdb=self.protein_file,
                        complex_pocket_pdb=self.protein_file,
                        ligand_crystal_pdb='',
                        ligand_crystal_mol2='',
                        ligand_crystal_sdf=ligand_file_sdf)
            data.pdb_id = ligand_name
  
        except:
            return None
        return data
  
    def __getitem__(self, idx):
        data = self._single_process(idx) 
        if data:
            start_index = None
            seq_order, seq_parent = BFS_search(data['ligand'].mol, start_atom_idx=start_index, random_seed=None)
            data['ligand'].seq_order = torch.tensor(seq_order[:], dtype=torch.long)
            data['ligand'].seq_parent = torch.tensor(seq_parent[:], dtype=torch.long)
        return data
    

def get_repeat_node(src_num, dst_num):
    return torch.arange(src_num, dtype=torch.long).repeat(dst_num), \
           torch.as_tensor(np.repeat(np.arange(dst_num), src_num), dtype=torch.long)


def generate_graph_4_Multi_PL(pocket_mol, pocket_atom_mol, complex_pocket_mol, ligand_pdb_mol, ligand_mol, frag_voc, use_rdkit_pos=False):
    torch.set_num_threads(1)
    # get pocket
    l_xyz =  torch.from_numpy(ligand_mol.GetConformer().GetPositions()).to(torch.float32)
    # get rdkit pos
    if use_rdkit_pos:
        rdkit_mol = mol2conformer(ligand_mol)
    else:
        rdkit_mol = ligand_mol
    # get feats
    p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s, p_node_name, _ , p_metal_mask, m_xyz, start_metal_mask, start_metal_index, start_ligand_index, start_ligand_indices = get_metalloprotein_feature(pocket_mol, pocket_atom_mol, complex_pocket_mol, ligand_pdb_mol)
    patom_xyz, pa_node_feature, pa_edge_index, pa_edge_feature, atom2res = get_patom_feature(pocket_atom_mol, allowed_res=p_node_name)
    l_rdkit_xyz, l_node_feature, l_edge_index, l_edge_feature, l_full_edge_s, l_interaction_edge_mask, l_cov_edge_mask, frag_smi_idxes, frag_node_s, frag_edge_idx, frag_edge_feature, atom2frag, l_coordinate_mask = get_ligand_feature(rdkit_mol, frag_voc)
    # to data
    data = HeteroData()
    # protein residue
    data.atom2res = atom2res = torch.tensor(atom2res, dtype=torch.long)
    data['protein'].node_s = p_node_s.to(torch.float32) 
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32) 
    data['protein'].xyz_full = p_xyz_full.to(torch.float32)    
    data['protein'].seq = p_seq.to(torch.int32)
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
    # protein atom
    data['protein_atom'].xyz = patom_xyz.to(torch.float32) 
    data['protein_atom'].node_s = pa_node_feature.to(torch.float32) 
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_index = pa_edge_index.to(torch.long)
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_s = pa_edge_feature.to(torch.float32) 
    # ligand
    data['ligand'].xyz = l_xyz.to(torch.float32)
    data['ligand'].pos = l_rdkit_xyz.to(torch.float32)
    data['ligand'].node_s = l_node_feature.to(torch.int32)
    data['ligand'].interaction_edge_mask = l_interaction_edge_mask
    data['ligand'].cov_edge_mask = l_cov_edge_mask
    data['ligand', 'l2l', 'ligand'].edge_index = l_edge_index.to(torch.long)
    data['ligand', 'l2l', 'ligand'].edge_s = l_edge_feature.to(torch.int32)
    data['ligand', 'l2l', 'ligand'].full_edge_s = l_full_edge_s.to(torch.float32)
    # frag
    data.atom2frag = torch.tensor(atom2frag, dtype=torch.long)
    data['frag'].node_s = frag_node_s.to(torch.int32)
    data['frag'].seq = frag_smi_idxes.to(torch.long)
    data['frag', 'f2f', 'frag'].edge_index = frag_edge_idx.to(torch.long)
    data['frag', 'f2f', 'frag'].edge_s = frag_edge_feature.to(torch.int32)
    data['ligand'].mol = rdkit_mol
    # protein-ligand
    data['protein', 'p2l', 'ligand'].edge_index = torch.stack(
        get_repeat_node(p_xyz.shape[0], l_xyz.shape[0]), dim=0)
    # metal
    data['metal'].lig_nos_mask = torch.tensor(l_coordinate_mask, dtype=torch.bool).view(-1)
    data['metal'].metal_mask = torch.tensor([[l and p for p in p_metal_mask] for l in l_coordinate_mask],dtype=torch.bool).view(-1) 
    data['metal'].xyz = m_xyz.to(torch.float32)
    data['metal'].start_metal_mask = torch.tensor([[l and p for p in start_metal_mask] for l in l_coordinate_mask],dtype=torch.bool).view(-1) 
    data['metal'].start_metal_index = torch.tensor(start_metal_index,dtype=torch.long)
    data['metal'].start_ligand_index = torch.tensor(start_ligand_index,dtype=torch.long)
    data['metal'].start_ligand_indices = torch.tensor(start_ligand_indices,dtype=torch.long)
    data['metal'].indices_group = torch.tensor([len(start_ligand_indices)],dtype=torch.long).unsqueeze(-1)
    if start_ligand_index != 9999:
        ligand_start_donar_mask = torch.zeros(l_coordinate_mask.shape, dtype=torch.bool).scatter_(0, torch.tensor(start_ligand_indices,dtype=torch.long), True)  
        data['metal'].start_donar_mask = torch.tensor([[l and p for p in start_metal_mask] for l in ligand_start_donar_mask],dtype=torch.bool).view(-1)   
    else:
        data['metal'].start_donar_mask = torch.tensor([])
    return data


def generate_graph_4_Multi_PL_vs(pocket_mol, pocket_atom_mol, complex_pocket_mol, ligand_mol, frag_voc, use_rdkit_pos=False):
    torch.set_num_threads(1)
    # get pocket
    l_xyz =  torch.from_numpy(ligand_mol.GetConformer().GetPositions()).to(torch.float32)
    # get rdkit pos
    if use_rdkit_pos:
        rdkit_mol = mol2conformer(ligand_mol)
    else:
        rdkit_mol = ligand_mol
    # get feats
    p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s, p_node_name, _ , p_metal_mask, m_xyz, start_metal_mask, start_metal_index = get_metalloprotein_feature_vs(pocket_mol, pocket_atom_mol)
    patom_xyz, pa_node_feature, pa_edge_index, pa_edge_feature, atom2res = get_patom_feature(pocket_atom_mol, allowed_res=p_node_name)
    l_rdkit_xyz, l_node_feature, l_edge_index, l_edge_feature, l_full_edge_s, l_interaction_edge_mask, l_cov_edge_mask, frag_smi_idxes, frag_node_s, frag_edge_idx, frag_edge_feature, atom2frag, l_coordinate_mask = get_ligand_feature(rdkit_mol, frag_voc)
    # to data
    data = HeteroData()
    # protein residue
    data.atom2res = atom2res = torch.tensor(atom2res, dtype=torch.long)
    data['protein'].node_s = p_node_s.to(torch.float32) 
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32) 
    data['protein'].xyz_full = p_xyz_full.to(torch.float32)    
    data['protein'].seq = p_seq.to(torch.int32)
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
    # protein atom
    data['protein_atom'].xyz = patom_xyz.to(torch.float32) 
    data['protein_atom'].node_s = pa_node_feature.to(torch.float32) 
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_index = pa_edge_index.to(torch.long)
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_s = pa_edge_feature.to(torch.float32) 
    # ligand
    data['ligand'].xyz = l_xyz.to(torch.float32)
    data['ligand'].pos = l_rdkit_xyz.to(torch.float32)
    data['ligand'].node_s = l_node_feature.to(torch.int32)
    data['ligand'].interaction_edge_mask = l_interaction_edge_mask
    data['ligand'].cov_edge_mask = l_cov_edge_mask
    data['ligand', 'l2l', 'ligand'].edge_index = l_edge_index.to(torch.long)
    data['ligand', 'l2l', 'ligand'].edge_s = l_edge_feature.to(torch.int32)
    data['ligand', 'l2l', 'ligand'].full_edge_s = l_full_edge_s.to(torch.float32)
    # frag
    data.atom2frag = torch.tensor(atom2frag, dtype=torch.long)
    data['frag'].node_s = frag_node_s.to(torch.int32)
    data['frag'].seq = frag_smi_idxes.to(torch.long)
    data['frag', 'f2f', 'frag'].edge_index = frag_edge_idx.to(torch.long)
    data['frag', 'f2f', 'frag'].edge_s = frag_edge_feature.to(torch.int32)
    data['ligand'].mol = rdkit_mol
    # protein-ligand
    data['protein', 'p2l', 'ligand'].edge_index = torch.stack(get_repeat_node(p_xyz.shape[0], l_xyz.shape[0]), dim=0)
    # metal
    data['metal'].lig_nos_mask = torch.tensor(l_coordinate_mask, dtype=torch.bool).view(-1)
    data['metal'].xyz = m_xyz.to(torch.float32)
    data['metal'].start_metal_mask = torch.tensor([[l and p for p in start_metal_mask] for l in l_coordinate_mask],dtype=torch.bool).view(-1)  
    data['metal'].start_metal_index = torch.tensor(start_metal_index,dtype=torch.long)
    return data


def mol2conformer(mol):
    m_mol = copy.deepcopy(mol)
    AllChem.Compute2DCoords(m_mol)
    return m_mol


def pdb2rdmol(pocket_pdb):
    pocket_atom_mol = Chem.MolFromPDBFile(pocket_pdb, removeHs=False)
    # pocket_atom_mol = Chem.RemoveAllHs(pocket_atom_mol)
    return pocket_atom_mol


def file2conformer(*args):
    for f in args:
        try:
            if os.path.splitext(f)[-1] == '.sdf':
                mol = Chem.MolFromMolFile(f, removeHs=True)
            else:
                mol = Chem.MolFromMol2File(f, removeHs=True)
            if mol is not None:
                mol = Chem.RemoveAllHs(mol)
                return mol
        except:
            continue
        

def get_graph_v1(pocket_pdb, complex_pocket_pdb, ligand_crystal_pdb, ligand_crystal_mol2='', ligand_crystal_sdf='', pocket_center=np.array([]), frag_voc={}):
    torch.set_num_threads(1)
    # get protein mol
    pocket_mol = mda.Universe(pocket_pdb)
    pocket_atom_mol = pdb2rdmol(pocket_pdb)
    complex_pocket_mol = mda.Universe(complex_pocket_pdb)
    ligand_pdb_mol = mda.Universe(ligand_crystal_pdb)  
    # get ligand_mol
    cry_ligand_mol = file2conformer(ligand_crystal_sdf, ligand_crystal_mol2)
    # generate graph
    hg = generate_graph_4_Multi_PL(pocket_mol, pocket_atom_mol, complex_pocket_mol, ligand_pdb_mol, cry_ligand_mol, frag_voc=frag_voc, use_rdkit_pos=False)
    return hg


def get_graph_vs(pocket_pdb, complex_pocket_pdb, ligand_crystal_pdb, ligand_crystal_mol2='', ligand_crystal_sdf='', pocket_center=np.array([]), frag_voc={}):
    torch.set_num_threads(1)
    # get protein mol
    pocket_mol = mda.Universe(pocket_pdb)
    pocket_atom_mol = pdb2rdmol(pocket_pdb)
    complex_pocket_mol = mda.Universe(complex_pocket_pdb)
    # get ligand_mol
    cry_ligand_mol = file2conformer(ligand_crystal_sdf, ligand_crystal_mol2)
    # generate graph
    hg = generate_graph_4_Multi_PL_vs(pocket_mol, pocket_atom_mol, complex_pocket_mol, cry_ligand_mol, frag_voc=frag_voc, use_rdkit_pos=False)
    return hg


def shuffle_center(xyz, noise=4):
    return xyz + torch.normal(mean=0, std=noise, size=(1, 3), dtype=torch.float) 
    

def random_rotation(xyz):
    random_rotation_matrix = torch.from_numpy(R.random().as_matrix()).to(torch.float32)
    lig_center = xyz.mean(dim=0)
    return (xyz - lig_center)@random_rotation_matrix.T + lig_center


def get_mol2_xyz_from_cmd(ligand_mol2):
    x = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}'" % ligand_mol2).read().splitlines()[1:-1]
    y = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}'" % ligand_mol2).read().splitlines()[1:-1]
    z = os.popen(
        "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}'" % ligand_mol2).read().splitlines()[1:-1]
    return np.asanyarray(list(zip(x, y, z))).astype(float)


def BFS_search(mol, start_atom_idx=None, random_seed=None):
    num_atom = mol.GetNumAtoms()
    if isinstance(random_seed, int):
        np.random.seed(random_seed) 
    if start_atom_idx is None:
        start_atom_idx = np.random.randint(0, num_atom)
    q = Queue()
    q.put(start_atom_idx)
    sorted_atom_indices = [start_atom_idx]
    visited = {i: False for i in range(num_atom)}
    visited[start_atom_idx] = True
    parent_lis = [9999]
    while not q.empty():
        current_atom_idx = q.get()
        for neighbor in mol.GetAtomWithIdx(current_atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if not visited[neighbor_idx]:
                q.put(neighbor_idx)
                visited[neighbor_idx] = True
                sorted_atom_indices.append(neighbor_idx)
                parent_lis.append(current_atom_idx)
    return sorted_atom_indices, parent_lis


if __name__ == '__main__':
    test_sdf = '/root/PDBBind/pdbbind2020/1a1e/1a1e_ligand_standard.sdf'
    mol = file2conformer(test_sdf)
    BFS_search(mol, start_atom_idx=0)
    

    