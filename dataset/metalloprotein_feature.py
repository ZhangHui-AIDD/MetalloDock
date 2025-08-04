#!usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch_cluster
import MDAnalysis as mda
from scipy.spatial import distance_matrix
from MDAnalysis.analysis import distances
import scipy.io as io
from rdkit import Chem


METAL = ["LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI","PT","ZN","CO","PD","AG","CR","FE","V","MN","HG",'GA', 
		"CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL","IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
		"TM","LU","HF","ZR","CE","U","PU","TH","6MO","FE2","MN3","CU1",'MO'] 
three_to_one_res = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': 'X'}
three2idx = {k:v for v, k in enumerate(['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'CO', 'CU', 
                                        'MG', 'SB', 'NI', 'V', 'PD', 'CS', 'CA', 'CE', 'PT', 
                                        'HG', 'NA', 'MN', 'AG', 'TL', 'RU', 'K', 'CD', 'ZN', 'LI', 'FE', 'EU', 'MO', 'UNK'])}                                       
three2self = {v:v for v in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'CO', 'CU', 
                                        'MG', 'SB', 'NI', 'V', 'PD', 'CS', 'CA', 'CE', 'PT', 
                                        'HG', 'NA', 'MN', 'AG', 'TL', 'RU', 'K', 'CD', 'ZN', 'LI', 'FE', 'EU', 'MO']}                           
three2self.update({'6MO':'MO', 'FE2':'FE', 'MN3':'MN', 'CU1':'CU'})
RES_MAX_NATOMS=24
Seq2Hydropathy = {'X': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
Seq2Volume = {'X': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
Seq2Charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZX'}}
Seq2Polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWVX"}}
Seq2Acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPVX"}}
Seq2Donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPVX"}}
metal2radius = {'CO':0.745, 'CU':0.73,  'MG': 0.72, 'SB':0.76, 'NI':0.69, 'V':0.59, 'PD':0.86, 'CS':1.67, 'CA':0.99, 'CE':1.034, 'PT':0.625, 'HG':1.02, 'NA':1.02, 'MN':0.46, 'AG':1.26, 'TL':1.5, 'RU':0.68, 'K':1.38, 'CD':0.97, 'ZN':0.74, 'LI':0.76, 'FE':0.645, 'EU':0.947, 'MO':0.65, 'UNK':0}

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def np_rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = np.linspace(D_min, D_max, D_count)
    # D_mu = D_mu.reshape((1, -1))
    D_sigma = (D_max - D_min) / D_count
    # D_expand = np.expand_dims(D, axis=-1)

    RBF = np.exp(-((D - D_mu) / D_sigma) ** 2)
    return RBF


def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0, 0]
    for idx, angle in enumerate([res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis


def check_connect(res_lis, i, j):
    if abs(i-j) == 1 and res_lis[i].segid == res_lis[j].segid:
        return 1
    else:
        return 0

def check_connect_metal(res_lis, i, j, coordination_edge):
    if abs(i-j) == 1 and res_lis[i].segid == res_lis[j].segid:
        return 1
    elif torch.any(torch.all(coordination_edge == torch.tensor([i, j]), dim=1)):
        return 2
    else:
        return 0


def positional_embeddings_v1(edge_index,
                                num_embeddings=16,
                                period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    # raw
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def calc_dist(res1, res2):
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array


def calculate_min_distance(metal_ion, ligand_ons_atoms, sub_indices):
    if ligand_ons_atoms.size(0) == 0:
        return float('inf'), None, None
    distances = torch.norm(ligand_ons_atoms - metal_ion, dim=1)
    min_distance, min_index = torch.min(distances, dim=0)
    return min_distance.item(), sub_indices[min_index.item()]


def obatin_edge(res_lis, src, dst):
    dist = calc_dist(res_lis[src], res_lis[dst])
    return dist.min()*0.1, dist.max()*0.1


def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(n, ca, c):
    c, n = _normalize(c - ca), _normalize(n - ca)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def check_res_exit(metal_coordination_edge_index, res_name2X_ca_id):   
    record_list = []                                      
    for pair in metal_coordination_edge_index:
        if len(set(pair).intersection(set(res_name2X_ca_id.keys()))) <1:
            record_list.append(pair)
    metal_coordination_protein_index = [edge for edge in metal_coordination_edge_index if edge not in record_list]
    return metal_coordination_protein_index


def find_matching_atom(complex_atom, ligand_universe, tolerance=0.1):
    """
    Find the corresponding atom in the ligand PDB file that matches the given atom in the complex.
    
    Args:
    - complex_atom: The atom from the complex structure
    - ligand_universe: The MDAnalysis Universe object for the ligand
    - tolerance: The distance tolerance for matching atom coordinates
    
    Returns:
    - The index of the matching atom in the ligand, or -1 if not found
    """
    complex_coords = complex_atom.position
    for ligand_atom in ligand_universe.atoms:
        ligand_coords = ligand_atom.position
        if np.linalg.norm(complex_coords - ligand_coords) < tolerance:
            return ligand_atom.index
    return -1


def get_mda_atom_coordinates(universe, id_lst):
    """
    Get the coordinates of atoms specified in id_lst as a tensor.

    Args:
    - universe: The MDAnalysis universe object containing the atoms.
    - id_lst: A list of atom IDs for which to get the coordinates.

    Returns:
    - A tensor containing the coordinates of the specified atoms.
    """
    id_lst = [i+1 for i in id_lst]
    selected_atoms = universe.atoms.select_atoms(f"id {' '.join(map(str, id_lst))}")   
    return torch.tensor(selected_atoms.positions, dtype=torch.float32)


def get_metalloprotein_feature(pocket_mol, mol, complex_mol, ligand_mol, top_k=30):
    with torch.no_grad():
        pure_res_lis, pure_res_key, pure_resname, seq, node_s, X_ca, X_n, X_c = [], [], [], [], [], [], [], []
        res_name2X_ca_id = {}
        metal_node_s, metal_pure_res_lis, metal_pure_res_key, metal_pure_resname, metal_seq, X_m, metal_coordination_edge_index, metal_ligand_ons_min_dist_lst, metal_ligand_ons_min_index_lst = [], [], [], [], [], [], [], [], []
        metal_name2X_ca_id, metal_id2donar_id = {},{}
        metal_cnt = 0

        for res in pocket_mol.residues:
            res_name = res.resname.strip()
            res_id = res.resid
            if res_name in three_to_one_res.keys():
                res_atoms = res.atoms
                dists = distances.self_distance_array(res_atoms.positions)
                ca = res_atoms.select_atoms("name CA")
                c = res_atoms.select_atoms("name C")
                n = res_atoms.select_atoms("name N")
                o = res_atoms.select_atoms("name O")
                if dists.size > 0:
                    dis_max = dists.max()
                    dis_min = dists.min()
                    dis_cao = distances.dist(ca,o)[-1][0]
                    dis_on = distances.dist(o,n)[-1][0]
                    dis_nc = distances.dist(n,c)[-1][0]
                    intra_dis = [dis_max*0.1, dis_min*0.1, dis_cao*0.1, dis_on*0.1, dis_nc*0.1] + \
                                    np_rbf(dis_max, D_min=0, D_max=20, D_count=16).tolist() + \
                                    np_rbf(dis_min, D_min=0, D_max=1.5, D_count=16).tolist() + \
                                    np_rbf(dis_cao, D_min=0, D_max=3.5, D_count=16).tolist() + \
                                    np_rbf(dis_on, D_min=0, D_max=4.5, D_count=16).tolist() + \
                                    np_rbf(dis_nc, D_min=0, D_max=3.5, D_count=16).tolist() 
                    seq_name_3 = three2self.get(res_name, 'UNK')
                    seq_name_1 = three_to_one.get(seq_name_3, 'X')
                    seq.append(three2idx[seq_name_3])
                    res_name2X_ca_id[f'{res_name}{res_id}'] = len(seq)-1          
                    X_ca.append(ca.positions[0])
                    X_n.append(n.positions[0])
                    X_c.append(c.positions[0])
                    res_feats = [Seq2Hydropathy[seq_name_1], Seq2Volume[seq_name_1] / 100, Seq2Charge[seq_name_1],
                    Seq2Polarity[seq_name_1], Seq2Acceptor[seq_name_1], Seq2Donor[seq_name_1], 0]
                    node_s.append(intra_dis+obtain_dihediral_angles(res)+res_feats)    
                    pure_res_lis.append(res)
                    pure_res_key.append(f'{res.segid}-{res.resid}{res.icode}')
                    pure_resname.append(f'{res.resname}{res.resid}{res.icode}')
            elif res_name in METAL:    
                for atom in res.atoms:
                    rdkit_metal_idx = atom.index
                    rdkit_metal_atom = mol.GetAtomWithIdx(int(rdkit_metal_idx))
                    atom_mass = rdkit_metal_atom.GetMass()
                    atom_charge = rdkit_metal_atom.GetFormalCharge()
                    #bland_metal_segid->SYSTEM
                    monomer_info = rdkit_metal_atom.GetMonomerInfo()
                    metal_segid = monomer_info.GetChainId().replace(' ', 'SYSTEM')
                metal_pure_res_lis.append(res)
                metal_pure_res_key.append(f'{metal_segid}-{res.resid}{res.icode}')
                metal_pure_resname.append(f'{res.resname}{res.resid}{res.icode}')
                seq_name_3 = three2self.get(res_name, 'UNK')
                metal_seq.append(three2idx[seq_name_3])
                metal_feats = [0 for _ in range(90)] + [(4/300) * math.pi * metal2radius[seq_name_3]**3, atom_charge, 0, 0, 0, atom_mass]
                metal_node_s.append(metal_feats)
                X_m.append(list(res.atoms.positions[0]))
                metal_name2X_ca_id[f'{res_name}{res_id}'] = len(X_m)-1
                if metal_segid == 'SYSTEM':
                    bond_metalres = complex_mol.select_atoms(f"resid {res_id}")                             
                else:
                    bond_metalres = complex_mol.select_atoms(f"segid {metal_segid} and resid {res_id}")     
                for atom in bond_metalres.atoms:   
                    for bonded_atom in atom.bonded_atoms:   
                        metal_coordination_edge_index.extend([[f'{bonded_atom.resname}{bonded_atom.resid}', f'{atom.resname}{atom.resid}'], \
                                                              [f'{atom.resname}{atom.resid}', f'{bonded_atom.resname}{bonded_atom.resid}']])
                        if bonded_atom.resname == ligand_mol.atoms[0].resname and bonded_atom.resid == ligand_mol.atoms[0].resid:
                            donar_index = find_matching_atom(bonded_atom, ligand_mol)
                            metal_id2donar_id.setdefault(metal_cnt, []).append(donar_index)
                if metal_id2donar_id:
                    metal_id2donar_id[metal_cnt] = list(set(metal_id2donar_id[metal_cnt]))
                    metal_ligand_ons_min_dist, metal_nearest_ligand_ons_idx = calculate_min_distance(res.atoms.positions[0], get_mda_atom_coordinates(ligand_mol,metal_id2donar_id[metal_cnt]), metal_id2donar_id[metal_cnt])
                    metal_ligand_ons_min_dist_lst.append(metal_ligand_ons_min_dist)      
                    metal_ligand_ons_min_index_lst.append(metal_nearest_ligand_ons_idx)
                    metal_cnt += 1
                                                                       
        pure_res_key.extend(metal_pure_res_key)
        metal_coordination_protein_index = check_res_exit(metal_coordination_edge_index, res_name2X_ca_id)   
        # generate metal ions masks
        metal_mask = torch.cat((torch.zeros(len(pure_res_lis), dtype=torch.bool), torch.ones(len(metal_pure_res_lis), dtype=torch.bool)))                                                   
        # generate start metal inos index masks
        start_metal_index_mask = torch.zeros(len(pure_res_lis)+len(metal_pure_res_lis), dtype=torch.bool)
        if metal_ligand_ons_min_dist_lst != []:            
            start_index = metal_ligand_ons_min_dist_lst.index(min(metal_ligand_ons_min_dist_lst))  
            start_ligand_index = metal_ligand_ons_min_index_lst[start_index]
            start_ligand_indices = metal_id2donar_id[start_index]
            start_metal_index = len(pure_res_lis)+start_index   
            start_metal_index_mask[start_metal_index] = True   
        else:
            start_metal_index, start_ligand_index, start_ligand_indices = len(pure_res_lis), 9999, [9999]
            start_metal_index_mask[start_metal_index] = True   
        # metal_seq_id to donar_id
        metal_id2donar_id = {key + len(pure_res_lis): value for key, value in metal_id2donar_id.items()}                  
        # node features + metal features
        seq += metal_seq
        seq = torch.from_numpy(np.asarray(seq))
        node_s += metal_node_s
        node_s = torch.from_numpy(np.asarray(node_s))
        # edge features
        metal_name2X_ca_id = {key: value + len(res_name2X_ca_id) for key, value in metal_name2X_ca_id.items()}
        res_name2X_ca_id.update(metal_name2X_ca_id)
        metal_coordination_protein_index = torch.tensor([[res_name2X_ca_id[pair[0]], res_name2X_ca_id[pair[1]]] for pair in metal_coordination_protein_index])
        pure_res_lis += metal_pure_res_lis
        # obtain the index of the metal in protein seq
        X_ca += X_m
        X_ca = torch.from_numpy(np.asarray(X_ca)) 	
        X_n += X_m
        X_n = torch.from_numpy(np.asarray(X_n))	
        X_c += X_m
        X_c = torch.from_numpy(np.asarray(X_c))	
        X_m = torch.from_numpy(np.asarray(X_m))
        X_center_of_mass = torch.from_numpy(pocket_mol.atoms.center_of_mass(compound='residues')) 
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
        dis_minmax = torch.from_numpy(np.asarray([obatin_edge(pure_res_lis, src, dst) for src, dst in edge_index.T])).view(edge_index.size(1), 2)
        dis_matx_center = distance_matrix(X_center_of_mass, X_center_of_mass)
        cadist = (torch.pairwise_distance(X_ca[edge_index[0]], X_ca[edge_index[1]]) * 0.1).view(-1,1)
        cedist = (torch.from_numpy(dis_matx_center[edge_index[0,:], edge_index[1,:]]) * 0.1).view(-1,1)
        if metal_coordination_protein_index.dim() > 1:
            edge_connect =  torch.from_numpy(np.asarray([check_connect_metal(pure_res_lis, x, y, metal_coordination_protein_index) for x,y in edge_index.T])).view(-1,1)
        else:
            edge_connect =  torch.from_numpy(np.asarray([check_connect(pure_res_lis, x, y) for x,y in edge_index.T])).view(-1,1)
        positional_embedding = positional_embeddings_v1(edge_index)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_s = torch.cat([edge_connect, cadist, cedist, dis_minmax, 
                            _rbf(cadist.view(-1), D_min=0, D_max=2.0, D_count=16, device='cpu'), 
                            _rbf(cedist.view(-1), D_min=0, D_max=2.0, D_count=16, device='cpu'), 
                            _rbf(dis_minmax[:, 0].view(-1), D_min=0, D_max=1.2, D_count=16, device='cpu'),
                            _rbf(dis_minmax[:, 1].view(-1), D_min=0, D_max=8, D_count=16, device='cpu'),
                            positional_embedding], dim=1)

        # vector features
        orientations = get_orientations(X_ca)
        orientations[-len(metal_seq),1] = torch.tensor([0,0,0])    
        orientations[-(len(metal_seq)+1),0] = torch.tensor([0,0,0])
        sidechains = get_sidechains(n=X_n, ca=X_ca, c=X_c)
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        xyz_full = torch.from_numpy(np.asarray([np.concatenate([res.atoms.positions[:RES_MAX_NATOMS, :], np.full((max(RES_MAX_NATOMS-len(res.atoms), 0), 3), np.nan)],axis=0) for res in pure_res_lis]))
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
        # full edge
        full_edge_s = torch.zeros((edge_index.size(1), 6))  # [s, d, t, f, non-cov, metal_coordination] 
        full_edge_s[edge_s[:, 0]==1, 0] = 1
        full_edge_s[edge_s[:, 0]==2, 5] = 1
        full_edge_s[edge_s[:, 0]==0, 4] = 1
        full_edge_s = torch.cat([full_edge_s, cadist], dim=-1)
        return (X_ca, xyz_full, seq, node_s, node_v, edge_index, edge_s, edge_v, full_edge_s, pure_res_key, 
                pure_resname, metal_mask, X_m, start_metal_index_mask, start_metal_index, start_ligand_index, start_ligand_indices)


def get_metalloprotein_feature_vs(pocket_mol, mol, top_k=30):
    with torch.no_grad():
        pure_res_lis, pure_res_key, pure_resname, seq, node_s, X_ca, X_n, X_c = [], [], [], [], [], [], [], []
        res_name2X_ca_id = {}
        metal_node_s, metal_pure_res_lis, metal_pure_res_key, metal_pure_resname, metal_seq, X_m, metal_coordination_edge_index, metal_ligand_ons_min_dist_lst, metal_ligand_ons_min_index_lst = [], [], [], [], [], [], [], [], []
        metal_name2X_ca_id, metal_id2donar_id = {},{}
        metal_cnt = 0

        for res in pocket_mol.residues:
            res_name = res.resname.strip()
            res_id = res.resid
            if res_name in three_to_one_res.keys():
                res_atoms = res.atoms
                dists = distances.self_distance_array(res_atoms.positions)
                ca = res_atoms.select_atoms("name CA")
                c = res_atoms.select_atoms("name C")
                n = res_atoms.select_atoms("name N")
                o = res_atoms.select_atoms("name O")
                if dists.size > 0:
                    dis_max = dists.max()
                    dis_min = dists.min()
                    dis_cao = distances.dist(ca,o)[-1][0]
                    dis_on = distances.dist(o,n)[-1][0]
                    dis_nc = distances.dist(n,c)[-1][0]
                    intra_dis = [dis_max*0.1, dis_min*0.1, dis_cao*0.1, dis_on*0.1, dis_nc*0.1] + \
                                    np_rbf(dis_max, D_min=0, D_max=20, D_count=16).tolist() + \
                                    np_rbf(dis_min, D_min=0, D_max=1.5, D_count=16).tolist() + \
                                    np_rbf(dis_cao, D_min=0, D_max=3.5, D_count=16).tolist() + \
                                    np_rbf(dis_on, D_min=0, D_max=4.5, D_count=16).tolist() + \
                                    np_rbf(dis_nc, D_min=0, D_max=3.5, D_count=16).tolist() 
                    seq_name_3 = three2self.get(res_name, 'UNK')
                    seq_name_1 = three_to_one.get(seq_name_3, 'X')
                    seq.append(three2idx[seq_name_3])
                    res_name2X_ca_id[f'{res_name}{res_id}'] = len(seq)-1           
                    X_ca.append(ca.positions[0])
                    X_n.append(n.positions[0])
                    X_c.append(c.positions[0])
                    res_feats = [Seq2Hydropathy[seq_name_1], Seq2Volume[seq_name_1] / 100, Seq2Charge[seq_name_1],
                    Seq2Polarity[seq_name_1], Seq2Acceptor[seq_name_1], Seq2Donor[seq_name_1], 0]
                    node_s.append(intra_dis+obtain_dihediral_angles(res)+res_feats)    
                    pure_res_lis.append(res)
                    pure_res_key.append(f'{res.segid}-{res.resid}{res.icode}')
                    pure_resname.append(f'{res.resname}{res.resid}{res.icode}')
            elif res_name in METAL:    
                for atom in res.atoms:
                    rdkit_metal_idx = atom.index
                    rdkit_metal_atom = mol.GetAtomWithIdx(int(rdkit_metal_idx))
                    atom_mass = rdkit_metal_atom.GetMass()
                    atom_charge = rdkit_metal_atom.GetFormalCharge()
                    monomer_info = rdkit_metal_atom.GetMonomerInfo()
                    metal_segid = monomer_info.GetChainId().replace(' ', 'SYSTEM')

                metal_pure_res_lis.append(res)
                metal_pure_res_key.append(f'{metal_segid}-{res.resid}{res.icode}')
                metal_pure_resname.append(f'{res.resname}{res.resid}{res.icode}')
                seq_name_3 = three2self.get(res_name, 'UNK')
                metal_seq.append(three2idx[seq_name_3])
                metal_feats = [0 for _ in range(90)] + [(4/300) * math.pi * metal2radius[seq_name_3]**3, atom_charge, 0, 0, 0, atom_mass]
                metal_node_s.append(metal_feats)
                X_m.append(list(res.atoms.positions[0]))
                metal_name2X_ca_id[f'{res_name}{res_id}'] = len(X_m)-1
                if metal_segid == 'SYSTEM':
                    bond_metalres = pocket_mol.select_atoms(f"resid {res_id}")                             
                else:
                    bond_metalres = pocket_mol.select_atoms(f"segid {metal_segid} and resid {res_id}")     
                for atom in bond_metalres.atoms:  
                    for bonded_atom in atom.bonded_atoms:   
                        metal_coordination_edge_index.extend([[f'{bonded_atom.resname}{bonded_atom.resid}', f'{atom.resname}{atom.resid}'], \
                                                              [f'{atom.resname}{atom.resid}', f'{bonded_atom.resname}{bonded_atom.resid}']])
                metal_cnt += 1                                             
        pure_res_key.extend(metal_pure_res_key)
        metal_coordination_protein_index = check_res_exit(metal_coordination_edge_index, res_name2X_ca_id)   
        # generate metal ions masks
        metal_mask = torch.cat((torch.zeros(len(pure_res_lis), dtype=torch.bool), torch.ones(len(metal_pure_res_lis), dtype=torch.bool)))                                                   
        # generate start metal inos index masks
        start_metal_index_mask = torch.zeros(len(pure_res_lis)+len(metal_pure_res_lis), dtype=torch.bool) 
        start_metal_index = len(pure_res_lis)
        start_metal_index_mask[start_metal_index] = True   
        seq += metal_seq
        seq = torch.from_numpy(np.asarray(seq))
        node_s += metal_node_s
        node_s = torch.from_numpy(np.asarray(node_s))
        # edge features
        metal_name2X_ca_id = {key: value + len(res_name2X_ca_id) for key, value in metal_name2X_ca_id.items()}
        res_name2X_ca_id.update(metal_name2X_ca_id)
        metal_coordination_protein_index = torch.tensor([[res_name2X_ca_id[pair[0]], res_name2X_ca_id[pair[1]]] for pair in metal_coordination_protein_index])
        pure_res_lis += metal_pure_res_lis
        # obtain the index of the metal in protein seq
        X_ca += X_m
        X_ca = torch.from_numpy(np.asarray(X_ca)) 	
        X_n += X_m
        X_n = torch.from_numpy(np.asarray(X_n))	
        X_c += X_m
        X_c = torch.from_numpy(np.asarray(X_c))	
        X_m = torch.from_numpy(np.asarray(X_m))
        X_center_of_mass = torch.from_numpy(pocket_mol.atoms.center_of_mass(compound='residues')) 
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
        dis_minmax = torch.from_numpy(np.asarray([obatin_edge(pure_res_lis, src, dst) for src, dst in edge_index.T])).view(edge_index.size(1), 2)
        dis_matx_center = distance_matrix(X_center_of_mass, X_center_of_mass)
        cadist = (torch.pairwise_distance(X_ca[edge_index[0]], X_ca[edge_index[1]]) * 0.1).view(-1,1)
        cedist = (torch.from_numpy(dis_matx_center[edge_index[0,:], edge_index[1,:]]) * 0.1).view(-1,1)
        if metal_coordination_protein_index.dim() > 1:
            edge_connect =  torch.from_numpy(np.asarray([check_connect_metal(pure_res_lis, x, y, metal_coordination_protein_index) for x,y in edge_index.T])).view(-1,1)
        else:
            edge_connect =  torch.from_numpy(np.asarray([check_connect(pure_res_lis, x, y) for x,y in edge_index.T])).view(-1,1)
        positional_embedding = positional_embeddings_v1(edge_index)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_s = torch.cat([edge_connect, cadist, cedist, dis_minmax, 
                            _rbf(cadist.view(-1), D_min=0, D_max=2.0, D_count=16, device='cpu'), 
                            _rbf(cedist.view(-1), D_min=0, D_max=2.0, D_count=16, device='cpu'), 
                            _rbf(dis_minmax[:, 0].view(-1), D_min=0, D_max=1.2, D_count=16, device='cpu'),
                            _rbf(dis_minmax[:, 1].view(-1), D_min=0, D_max=8, D_count=16, device='cpu'),
                            positional_embedding], dim=1)

        # vector features
        orientations = get_orientations(X_ca)
        orientations[-len(metal_seq),1] = torch.tensor([0,0,0])    
        orientations[-(len(metal_seq)+1),0] = torch.tensor([0,0,0])
        sidechains = get_sidechains(n=X_n, ca=X_ca, c=X_c)
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        xyz_full = torch.from_numpy(np.asarray([np.concatenate([res.atoms.positions[:RES_MAX_NATOMS, :], np.full((max(RES_MAX_NATOMS-len(res.atoms), 0), 3), np.nan)],axis=0) for res in pure_res_lis]))
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
        # full edge
        full_edge_s = torch.zeros((edge_index.size(1), 6))  # [s, d, t, f, non-cov, metal_coordination]                                              
        full_edge_s[edge_s[:, 0]==1, 0] = 1
        full_edge_s[edge_s[:, 0]==2, 5] = 1
        full_edge_s[edge_s[:, 0]==0, 4] = 1
        full_edge_s = torch.cat([full_edge_s, cadist], dim=-1)
        return (X_ca, xyz_full, seq, node_s, node_v, edge_index, edge_s, edge_v, full_edge_s, pure_res_key, 
                pure_resname, metal_mask, X_m, start_metal_index_mask, start_metal_index)
    

if __name__ == '__main__':
    pass
