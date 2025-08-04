#!usr/bin/env python3
# -*- coding:utf-8 -*-
from copy import deepcopy
import torch
import networkx as nx
import numpy as np
import copy
from torch_geometric.utils import to_networkx
from rdkit import Chem
import warnings
from rdkit.Chem import AllChem
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.spatial import distance_matrix


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


# orderd by perodic table
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))
bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol dim=18
        
        GetChiralTag(): one-hot embedding for atomic chiral tag dim=5
        
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs dim=5
        
        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
        
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom 
        
        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom
        
        GetHybridization(): one-hot embedding for the atom's hybridization
        
        GetIsAromatic(): whether the atom is aromatic
        
        IsInRing(): whether the atom is in a ring
        18 + 5 + 8 + 12 + 8 + 9 + 10 + 9 + 3 + 4 
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab, allow_unknown=True) + \
           onehot(atom.GetHybridization(), hybridization_vocab, allow_unknown=True) + \
            onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic(), atom.IsInRing(), atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]+[atom.IsInRingSize(i) for i in range(3, 7)]


def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond
        
        GetBondDir(): one-hot embedding for the direction of the bond
        
        GetStereo(): one-hot embedding for the stereo configuration of the bond
        
        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return onehot(bond.GetBondType(), bond_type_vocab, allow_unknown=True) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab, allow_unknown=True) + \
           [int(bond.GetIsConjugated())]


def get_full_connected_edge(frag):
    frag = np.asarray(list(frag))
    return torch.from_numpy(np.repeat(frag, len(frag)-1)), \
        torch.from_numpy(np.concatenate([np.delete(frag, i) for i in range(frag.shape[0])], axis=0))


def remove_repeat_edges(new_edge_index, refer_edge_index, N_atoms):
    new = to_dense_adj(new_edge_index, max_num_nodes=N_atoms)
    ref = to_dense_adj(refer_edge_index, max_num_nodes=N_atoms)
    delta_ = new - ref
    delta_[delta_<1] = 0
    unique, _ = dense_to_sparse(delta_)
    return unique


def get_patom_feature(mol, allowed_res, use_chirality=True):
    '''
    for bio-molecule
    '''
    xyz = mol.GetConformer().GetPositions()
    # covalent
    N_atoms = mol.GetNumAtoms()
    node_feature = []
    edge_index = []
    edge_feature = []
    not_allowed_node = []
    atom2res = []
    graph_idx_2_mol_idx = {}
    mol_idx_2_graph_idx = {}
    graph_idx = 0
    coords = []
    # atom
    for idx, atom in enumerate(mol.GetAtoms()):
        monomer_info = atom.GetMonomerInfo()
        node_name = f"{monomer_info.GetChainId().replace(' ', 'SYSTEM')}-{monomer_info.GetResidueNumber()}{monomer_info.GetInsertionCode().strip()}"
        try:
            atom2res.append(allowed_res.index(node_name))
        except:
            not_allowed_node.append(idx)
            continue
        # node
        coords.append(xyz[idx])
        node_feature.append(atom_default(atom))
        graph_idx_2_mol_idx[graph_idx] = idx
        mol_idx_2_graph_idx[idx] = graph_idx
        # update graph idx
        graph_idx += 1
    # bond
    for bond in mol.GetBonds():
        src_idx, dst_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            src_idx = mol_idx_2_graph_idx[src_idx]
            dst_idx = mol_idx_2_graph_idx[dst_idx]
        except:
            continue
        edge_index.append([src_idx, dst_idx])
        edge_index.append([dst_idx, src_idx])
        edge_feats = bond_default(bond)
        edge_feature.append(edge_feats)
        edge_feature.append(edge_feats)
    # to tensor
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    N_atoms = len(mol_idx_2_graph_idx)
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            try:
                i = mol_idx_2_graph_idx[i]
            except:
                continue
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature,chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    xyz = torch.from_numpy(np.asanyarray(coords)).float()
    x = (xyz, node_feature, edge_index, edge_feature, atom2res) 
    assert node_feature.size(0) > 10, 'pocket rdkit mol fail'
    return x 

def get_metal_coordinate_mask(element):
    metal_coordinate_atom_type = ['O', 'N', 'S']
    if element in metal_coordinate_atom_type:
        return True
    else:
        return False
    
def get_ligand_feature(mol, frag_voc, use_chirality=True):
    xyz = mol.GetConformer().GetPositions()
    # covalent
    N_atoms = mol.GetNumAtoms()
    node_feature = []
    edge_index = []
    edge_feature = []
    G = nx.Graph()
    G_rb_idx = []
    ele = []
    metal_coordinate_mask = []             
    for idx, atom in enumerate(mol.GetAtoms()):
        ele.append(atom.GetSymbol())
        metal_coordinate_mask.append(get_metal_coordinate_mask(atom.GetSymbol()))       
        # node
        node_feature.append(atom_default(atom))
        # edge
        for bond in atom.GetBonds():
            edge_feature.append(bond_default(bond))
            for bond_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                if bond_idx != idx:  
                    edge_index.append([idx, bond_idx])
                    G.add_edge(idx, bond_idx)
                    if bond_idx > idx and bond.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond.IsInRing():
                        G_rb_idx.append((idx, bond_idx))
                        
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()  # nodes_chemical_features
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature,chiral_arr.float()],dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    cov_edge_num = edge_index.size(1)
    # 0:4 bond type 5:11 bond direction 12:18 bond stero 19 bond conjunction
    l_full_edge_s = edge_feature[:, :5]
    xyz = torch.from_numpy(xyz)
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    # get fragments based on rotation bonds
    frags = []
    atom2frag = np.zeros(N_atoms)
    # for e in G.edges():
    for e in G_rb_idx:
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        for f in nx.connected_components(G2):
            frags.append(f)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [list(i) for i in frags if i != set()]
    else:
        frags = [list(range(N_atoms))]
    # frag edge
    atom_in_frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
    edge_index_new = remove_repeat_edges(new_edge_index=atom_in_frag_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], torch.pairwise_distance(xyz[edge_index_new[0]], xyz[edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_atoms, N_atoms)) - torch.eye(N_atoms, N_atoms)
    interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    edge_index_new = remove_repeat_edges(new_edge_index=interaction_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    interaction_edge_mask = torch.ones((edge_feature.size(0),))
    interaction_edge_mask[-edge_feature_new.size(0):] = 0
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], -torch.ones(edge_feature_new.size(0), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # cov edge mask
    cov_edge_mask = torch.zeros(edge_feature.size(0),)
    cov_edge_mask[:cov_edge_num] = 1
    #### frag graph 
    # get atom2fragidx
    frag_smi_idxes = []
    frag_node_s = []
    for idx, f in enumerate(frags):
        atom2frag[f] = idx 
        frag_smi_idxes.append(frag_voc.get(Chem.MolFragmentToSmiles(mol, f), len(frag_voc)))
        frag_poses = xyz[f]
        frag_intra_dis = distance_matrix(frag_poses, frag_poses)
        if len(f)>1:
            frag_intra_dis = frag_intra_dis[~np.eye(len(frag_poses),dtype=bool)] 
        dis_max = frag_intra_dis.max()
        dis_min = frag_intra_dis.min()
        frag_center_mass_xyz = frag_poses.mean(axis=0)
        frag_node_s.append([dis_max*0.1, dis_min*0.1,] + np_rbf(dis_max, D_min=0, D_max=20, D_count=16).tolist() + np_rbf(dis_min, D_min=0, D_max=1.5, D_count=16).tolist()) 
    # frag edge index
    frag_edge_idx = []
    frag_edge_feature = []
    for (u, v) in G_rb_idx:
        frag_edge_idx.append([atom2frag[u], atom2frag[v]])
        frag_edge_idx.append([atom2frag[v], atom2frag[u]])
        frag_edge_feature.append(bond_default(mol.GetBondBetweenAtoms(u, v)))
        frag_edge_feature.append(bond_default(mol.GetBondBetweenAtoms(v, u)))
    # to tensor
    frag_smi_idxes = torch.from_numpy(np.asanyarray(frag_smi_idxes)).long()
    frag_node_s = torch.from_numpy(np.asanyarray(frag_node_s)).float()
    frag_edge_feature = torch.from_numpy(np.asanyarray(frag_edge_feature)).float()
    frag_edge_idx = torch.from_numpy(np.asanyarray(frag_edge_idx).T).long()
    l_full_edge_s = torch.cat((l_full_edge_s[:, :5], torch.zeros(l_full_edge_s.shape[0], 1), l_full_edge_s[:, 5:]), dim=-1)  
    metal_coordinate_mask = torch.tensor(metal_coordinate_mask, dtype=torch.bool)  
    x = (xyz, node_feature, edge_index, edge_feature, l_full_edge_s, interaction_edge_mask.bool(), cov_edge_mask.bool(), frag_smi_idxes, frag_node_s, frag_edge_idx, frag_edge_feature, atom2frag, metal_coordinate_mask) 
    return x 


if __name__ == '__main__':
    pass