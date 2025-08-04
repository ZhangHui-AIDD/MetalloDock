#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from architecture.GVP_Block import GVP_embedding
from architecture.GraphTransformer_Block import GraghTransformer
from architecture.MDN_Block import MDN_Block
from architecture.EGNN_Block import EGNN
from utils.fns import mdn_loss_fn, calculate_probablity, BFS_search 
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import to_dense_batch
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from itertools import chain
import time
from queue import Queue
print = partial(print, flush=True)

class MetalloDock(nn.Module):
    def __init__(self, hierarchical=True):
        super(MetalloDock, self).__init__()
        self.hierarchical = hierarchical
        # encoders
        self.lig_encoder = GraghTransformer(
            in_channels=89, 
            edge_features=20, 
            num_hidden_channels=128,
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )
        self.pro_encoder = GVP_embedding(
            (96, 3), (128, 16), (85, 1), (32, 1), seq_in=True, vector_gate=True)                 
        if hierarchical:
            self.frag_embedding = nn.Embedding(1101, 128)
            self.frag_encoder = GraghTransformer(
                in_channels=34+128, 
                edge_features=20, 
                num_hidden_channels=128,
                activ_fn=torch.nn.SiLU(),
                transformer_residual=True,
                num_attention_heads=4,
                norm_to_apply='batch',
                dropout_rate=0.15,
                num_layers=6)
            self.proatom_encoder = GraghTransformer(
                in_channels=89, 
                edge_features=20, 
                num_hidden_channels=128,
                activ_fn=torch.nn.SiLU(),
                transformer_residual=True,
                num_attention_heads=4,
                norm_to_apply='batch',
                dropout_rate=0.15,
                num_layers=6)
            # atom to protein
            self.merge_hierarchical_a2r = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
            self.graph_norm_hierarchical_a2r = GraphNorm(in_channels=128)
            self.merge_hierarchical_r2a = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
            self.graph_norm_hierarchical_r2a = GraphNorm(in_channels=256)
            # frag to ligand atom
            self.merge_hierarchical_f2a = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
            self.graph_norm_hierarchical_f2a = GraphNorm(in_channels=256)
        # graph norm
        self.graph_norm = GraphNorm(128)
        # interaction graph
        self.edge_s_init_layer = nn.Linear(in_features=7, out_features=128)
        # pose prediction
        self.pose_sampling_layers = nn.ModuleList([EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15, update_pos=False) if i < 7 else EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15, update_pos=True) for i in range(8) ])
        # scoring 
        self.scoring_layers = MDN_Block(hidden_dim=128, 
                                         n_gaussians=10, 
                                        dropout_rate=0.10, 
                                        dist_threhold=7.)
        self.bce_loss = nn.BCELoss()


    def cal_rmsd(self, pos_ture, pos_pred, batch, if_r=True):
        if if_r:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch).sqrt()
        else:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch)
    
    
    def forward(self, data, pos_r, start_rate):
        # mdn aux labels
        atom_types_label = torch.argmax(data['ligand'].node_s[:,:18], dim=1, keepdim=False)
        bond_types_label = torch.argmax(data['ligand', 'l2l', 'ligand'].edge_s[data['ligand'].cov_edge_mask][:, :5], dim=1, keepdim=False)
        # encoder 
        pro_node_s, _, lig_node_s, data, pro_nodes_num = self.encoder(data)
        # predict distance distribution 
        pi, sigma, mu, _, atom_types, bond_types, coor_donar_probability, C_mask, B, N_l = self.scoring_layers(lig_s=lig_node_s, lig_batch=data['ligand'].batch, 
                                                                                         pro_s=pro_node_s, pro_batch=data['protein'].batch,
                                                                                         donar_metal_mask = data['metal'].start_metal_mask,
                                                                                         edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        dist, dist_ca = self.scoring_layers.cal_pair_dist_and_select_nearest_residue(lig_pos=data['ligand'].xyz, lig_batch=data['ligand'].batch, 
                                                              pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch)
        # EGNN
        if pos_r:
            if torch.rand(1).item() < start_rate:
                # graph norm through a interaction graph
                data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch = self.construct_interaction_graph(data, pro_nodes_num, train=True)
                pos, pred_xyz, true_xyz, sample_batch = self.sampling_atom_pos(data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=True)
                # loss     
                rmsd_losss = self.cal_rmsd(pos_ture=true_xyz, pos_pred=pred_xyz, batch=sample_batch, if_r=True)
            else:
                # graph norm through a interaction graph
                data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch = self.construct_interaction_graph(data, pro_nodes_num, train=True)
                pos, pred_xyz, true_xyz, sample_batch = self.sampling_atom_pos(data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=True)
                # loss     
                rmsd_losss = self.cal_rmsd(pos_ture=true_xyz, pos_pred=pred_xyz, batch=sample_batch, if_r=True)
        else:
            rmsd_losss = torch.zeros(1, device=data['protein'].node_s.device, dtype=torch.float)
        # mdn block
        aux_r = 0.001
        # scoring loss
        mdn_loss_true = mdn_loss_fn(pi, sigma, mu, dist)[torch.where(dist <= self.scoring_layers.dist_threhold)[0]].mean().float() 
        # adding aux loss
        mdn_loss_true = mdn_loss_true + aux_r*F.cross_entropy(atom_types, atom_types_label) + aux_r*F.cross_entropy(bond_types, bond_types_label) 
        # predict coordinated donar loss
        coor_donar_truth = data['metal'].start_donar_mask.to(torch.float32)
        metal_coor_loss = self.bce_loss(coor_donar_probability.to(torch.float32), coor_donar_truth[data['metal'].start_metal_mask])
        return rmsd_losss, mdn_loss_true, metal_coor_loss, coor_donar_probability.to(torch.float32), coor_donar_truth[data['metal'].start_metal_mask], data['metal'].start_metal_mask
    
    
    def encoder(self, data):
        # encoder
        lig_node_s = self.lig_encoder(data['ligand'].node_s.to(torch.float32), 
                                      data['ligand', 'l2l', 'ligand'].edge_s[data['ligand'].cov_edge_mask].to(torch.float32), 
                                      data['ligand', 'l2l', 'ligand'].edge_index[:,data['ligand'].cov_edge_mask])
        pro_node_s = self.pro_encoder((data['protein']['node_s'], data['protein']['node_v']),
                                                    data[("protein", "p2p", "protein")]["edge_index"],
                                                    (data[("protein", "p2p", "protein")]["edge_s"],
                                                    data[("protein", "p2p", "protein")]["edge_v"]),
                                                    data['protein'].seq)
        # hierarchical
        if self.hierarchical:
            # protein atom to residue
            proatom_node_s = self.proatom_encoder(data['protein_atom'].node_s.to(torch.float32), data['protein_atom', 'pa2pa', 'protein_atom'].edge_s.to(torch.float32), data['protein_atom', 'pa2pa', 'protein_atom'].edge_index)
            proatom_node_s_ = scatter(proatom_node_s, index=data.atom2res, reduce='sum', dim=0)
            proatom_node_s_ = self.graph_norm_hierarchical_a2r(proatom_node_s_, data['protein'].batch)
            pro_node_s_ = torch.cat([pro_node_s, proatom_node_s_], dim=-1)
            pro_node_s = self.merge_hierarchical_a2r(pro_node_s_)
            # ligand frag to atom
            data['frag'].node_s = torch.cat([data['frag'].node_s.to(torch.float32), self.frag_embedding(data['frag'].seq)], dim=1)
            ligfrag_node_s = self.frag_encoder(
                data['frag'].node_s, 
                data['frag', 'f2f', 'frag'].edge_s.to(torch.float32), 
                data['frag', 'f2f', 'frag'].edge_index)
            lig_node_s = torch.cat([lig_node_s, ligfrag_node_s[data.atom2frag]], dim=-1)
            lig_node_s = self.graph_norm_hierarchical_f2a(lig_node_s, data['ligand'].batch)
            lig_node_s = self.merge_hierarchical_f2a(lig_node_s)
        # graph norm through a protein-ligand graph
        pro_nodes_num = data['protein'].num_nodes
        node_s = self.graph_norm(torch.cat([pro_node_s, lig_node_s], dim=0), torch.cat([data['protein'].batch, data['ligand'].batch], dim=-1))
        data['protein'].node_s, data['ligand'].node_s = node_s[:pro_nodes_num], node_s[pro_nodes_num:]
        return pro_node_s, proatom_node_s, lig_node_s, data, pro_nodes_num
    

    def construct_interaction_graph(self, data, pro_nodes_num, train=True):       
        batch_seq_order, _ = to_dense_batch(data['ligand'].seq_order, batch=data['ligand'].batch, fill_value=9999)
        batch_seq_parent, _ = to_dense_batch(data['ligand'].seq_parent, batch=data['ligand'].batch, fill_value=9999)
        #batch_seq_order = batch_seq_order[:, 1:]                    
        #batch_seq_parent = batch_seq_parent[:, 1:]
        # transform lig batch idx to pytorch geometric batch idx
        lig_nodes_num_per_sample = torch.unique(data['ligand'].batch, return_counts=True)[1]
        lig_nodes_num_per_sample = torch.cumsum(lig_nodes_num_per_sample, dim=0)
        batch_seq_order += pro_nodes_num
        batch_seq_order[1:] += lig_nodes_num_per_sample[:-1].view((-1, 1)) 
        batch_seq_parent[:,1:] += torch.cat([torch.zeros((1,1)).to(lig_nodes_num_per_sample.device), lig_nodes_num_per_sample[:-1].view((-1, 1))], dim=0).to(torch.long) + pro_nodes_num
        pro_nodes_num_per_sample = torch.unique(data['protein'].batch, return_counts=True)[1]
        pro_nodes_num_per_sample = torch.cumsum(pro_nodes_num_per_sample, dim=0)
        batch_seq_parent[1:,0] += pro_nodes_num_per_sample[:-1]
        # construct interaction graph
        batch = torch.cat([data['protein'].batch, data['ligand'].batch], dim=-1)
        # edge index
        u = torch.cat([
            data[("protein", "p2p", "protein")]["edge_index"][0], 
            data[('ligand', 'l2l', 'ligand')]["edge_index"][0]+pro_nodes_num, 
            data[('protein', 'p2l', 'ligand')]["edge_index"][0], data[('protein', 'p2l', 'ligand')]["edge_index"][1]+pro_nodes_num], dim=-1)
        v = torch.cat([
            data[("protein", "p2p", "protein")]["edge_index"][1], 
            data[('ligand', 'l2l', 'ligand')]["edge_index"][1]+pro_nodes_num, 
            data[('protein', 'p2l', 'ligand')]["edge_index"][1]+pro_nodes_num, data[('protein', 'p2l', 'ligand')]["edge_index"][0]], dim=-1)
        edge_index = torch.stack([u, v], dim=0)
        # pose
        xyz = torch.cat([data['protein'].xyz, data['ligand'].xyz], dim=0)
        dist = torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]], keepdim=True)
        # features
        node_s = torch.cat([data['protein'].node_s, data['ligand'].node_s], dim=0)
        edge_s = torch.zeros((data[('protein', 'p2l', 'ligand')]["edge_index"][0].size(0)*2, 7), device=node_s.device)
        edge_s[:, -1] = -1
        edge_s = torch.cat([data[("protein", "p2p", "protein")].full_edge_s, data['ligand', 'l2l', 'ligand'].full_edge_s, edge_s], dim=0)
        edge_s = self.edge_s_init_layer(edge_s)
        # generation mask
        generate_mask_from_protein = torch.ones((node_s.size(0), 1), device=node_s.device)
        generate_mask_from_protein[pro_nodes_num:] = 0
        generate_mask_from_protein[batch_seq_parent[:, 0]] = 1
        # for random masking in training process
        generate_mask_from_lig = torch.ones((node_s.size(0), 1), device=node_s.device)
        if train:
            # random choose six consecutive nodes for model training
            num_consecutive_nodes = 5
            # get node number for each sample in a batch. minus 1 because the first node is focal residue and not included in the random selection
            num_nodes_per_batch = torch.unique(data['ligand'].batch, return_counts=True)[1] - 1
            # random select start index for each sample in a batch; start from 0
            start_indices = torch.clamp(num_nodes_per_batch - num_consecutive_nodes, min=0)
            start_indices = (torch.clamp(torch.randn(size=start_indices.shape), min=0, max=0).to(start_indices.device) * start_indices).long() 
            # get end index for each sample in a batch
            end_indices = start_indices + num_consecutive_nodes
            # get batch select index
            batch_select_index = torch.arange(batch_seq_order.size(1)).unsqueeze(0).to(start_indices.device)
            # get six consecutive nodes index for each sample in a batch
            predict_batch_select_index = (batch_select_index >= start_indices.unsqueeze(1)) & (batch_select_index < end_indices.unsqueeze(1))
            # get nodes with masked coordinates for each sample in a batch
            start_batch_select_index = (batch_select_index >= start_indices.unsqueeze(1)) 
            start_idx = batch_seq_order[start_batch_select_index]
            start_idx = start_idx[start_idx < pro_nodes_num + 9999]
            # mask nodes
            generate_mask_from_lig[start_idx] = 0
            batch_seq_order = batch_seq_order[predict_batch_select_index].view((-1, num_consecutive_nodes))
            batch_seq_parent = batch_seq_parent[predict_batch_select_index].view((-1, num_consecutive_nodes))
        # to data
        data.edge_index, data.edge_s, data.node_s, data.dist = edge_index, edge_s, node_s, dist
        return data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch
    

    def sampling_atom_pos(self, data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=True):
        pred_xyz = []
        true_xyz = []
        sample_batch = []
        pos = xyz.clone()
        # add noise to xyz for generation stability
        if train:
            pos[pro_nodes_num:] = torch.randn_like(pos[pro_nodes_num:])*0.3 + pos[pro_nodes_num:]
        generate_mask = generate_mask_from_lig if train else generate_mask_from_protein
        for r_idx in range(batch_seq_order.size(1)):
            # get node_s, edge_index, edge_s
            node_s, edge_index, edge_s, = data.node_s, data.edge_index, data.edge_s
            # get parent node idxes and generate node idxes
            parent_node_idxes = batch_seq_parent[:, r_idx]
            generate_node_idxes = batch_seq_order[:, r_idx]
            parent_node_idxes = parent_node_idxes[parent_node_idxes < pro_nodes_num + 9999]
            generate_node_idxes = generate_node_idxes[generate_node_idxes < pro_nodes_num + 9999]
            generate_node_dist = torch.pairwise_distance(xyz[parent_node_idxes], xyz[generate_node_idxes], keepdim=True)
            ### mask unkonwn distance
            generate_mask[generate_node_idxes] = 1
            mask_inv = (generate_mask == 0) 
            mask_edge_inv = mask_inv[edge_index[0]] | mask_inv[edge_index[1]] 
            # init next node pos
            pos_ = pos.clone()
            pos_[generate_node_idxes] = pos_[parent_node_idxes] + torch.clamp(torch.randn_like(pos_[parent_node_idxes])*0.1, min=-0.1, max=0.1)
            # egnn layers for pose prediction
            for idx, layer in enumerate(self.pose_sampling_layers):
                node_s, edge_s, pred_pose = layer(node_s, edge_s, edge_index, generate_node_dist, pos_, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num, batch)
            # for training
            sample_batch.append(data['ligand'].batch[generate_node_idxes-pro_nodes_num])
            true_xyz.append(xyz[generate_node_idxes, :])
            pred_xyz.append(pred_pose)
            # update vector feats
            if not train:
                # update pos
                pos[generate_node_idxes] = pred_pose
                # c_pos[generate_node_idxes] = pred_pose
        return pos, torch.cat(pred_xyz, dim=0), torch.cat(true_xyz, dim=0), torch.cat(sample_batch, dim=0)
    
    
    def scoring_PL_conformation(self, pi, sigma, mu, dist, dist_threhold, c_batch, batch_size):
        '''
        scoring the protein-ligand binding strength
        '''
        mdn_score = calculate_probablity(pi, sigma, mu, dist)
        mdn_score[torch.where(dist > dist_threhold)[0]] = 0.
        mdn_score = scatter(mdn_score, index=c_batch, dim=0, reduce='sum', dim_size=batch_size).float()
        return mdn_score

    def scoring_PL_complex(self, data, dist_threhold=5.):
        # mdn aux labels
        batch_size = data['ligand'].batch.max().item() + 1
        # encoder 
        pro_node_s, _, lig_node_s, data, pro_nodes_num = self.encoder(data)
        # predict distance distribution and focal residue
        pi, sigma, mu, c_batch, atom_types, bond_types, coor_donar_probability, C_mask, B, N_l = self.scoring_layers(lig_s=lig_node_s, lig_batch=data['ligand'].batch,
                                                                                         pro_s=pro_node_s, pro_batch=data['protein'].batch,
                                                                                         donar_metal_mask = data['metal'].start_metal_mask,
                                                                                         edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        pos = data['ligand'].xyz
        lig_pose = pos
        # scoring
        dist, dist_ca = self.scoring_layers.cal_pair_dist_and_select_nearest_residue_4vs(lig_pos=lig_pose, lig_batch=data['ligand'].batch, 
                                                            pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch, C_mask=C_mask, B=B, N_l=N_l)
        # scoring loss
        score = self.scoring_PL_conformation(pi, sigma, mu, dist, dist_threhold, c_batch, batch_size)
        return score

    def ligand_docking(self, data, docking=True, scoring=True, dist_threhold=5.):
        # mdn aux labels
        batch_size = data['ligand'].batch.max().item() + 1
        # encoder 
        pro_node_s, _, lig_node_s, data, pro_nodes_num = self.encoder(data)
        # predict distance distribution and focal residue
        pi, sigma, mu, c_batch, atom_types, bond_types, coor_donar_probability, C_mask, B, N_l = self.scoring_layers(lig_s=lig_node_s, lig_batch=data['ligand'].batch,
                                                                                         pro_s=pro_node_s, pro_batch=data['protein'].batch,
                                                                                         donar_metal_mask = data['metal'].start_metal_mask,
                                                                                         edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        
        if docking:
            # select the focal residue
            # graph norm through a interaction graph
            data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch = self.construct_interaction_graph(data, pro_nodes_num, train=False)
            pos, pred_xyz, true_xyz, sample_batch = self.sampling_atom_pos(data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=False)
            # # loss     
            # rmsd_losss = self.cal_rmsd(pos_ture=true_xyz, pos_pred=pred_xyz, batch=sample_batch, if_r=True)
            # print('rmsd_losss', rmsd_losss.mean().item())
        else:
            pos = xyz
        lig_pose = pos[pro_nodes_num:]
        # scoring
        if scoring:
            dist, dist_ca = self.scoring_layers.cal_pair_dist_and_select_nearest_residue(lig_pos=lig_pose, lig_batch=data['ligand'].batch, 
                                                              pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch)
            # scoring loss
            score = self.scoring_PL_conformation(pi, sigma, mu, dist, dist_threhold, c_batch, batch_size)
        else:
            score = torch.zeros(batch_size, 1)
        return lig_pose, score


    def ligand_docking_coordination(self, data, docking=True, scoring=True, dist_threhold=5.):
        my_device = data['metal'].xyz.device
        # mdn aux labels
        batch_size = data['ligand'].batch.max().item() + 1
        # encoder 
        pro_node_s, _, lig_node_s, data, pro_nodes_num = self.encoder(data)
        # predict distance distribution and focal residue
        pi, sigma, mu, c_batch, atom_types, bond_types, coor_donar_probability, C_mask, B, N_l = self.scoring_layers(lig_s=lig_node_s, lig_batch=data['ligand'].batch,
                                                                                         pro_s=pro_node_s, pro_batch=data['protein'].batch,
                                                                                         donar_metal_mask = data['metal'].start_metal_mask,
                                                                                         edge_index=data['ligand', 'l2l', 'ligand'].edge_index[:, data['ligand'].cov_edge_mask])
        # predict start ligand index          
        original_tensor = torch.zeros(data['metal'].lig_nos_mask.shape, dtype=coor_donar_probability.dtype, device=my_device) 
        original_tensor[data['metal'].lig_nos_mask] = coor_donar_probability  
        original_batch = torch.bincount(data['ligand'].batch)  
        original_start_indices = torch.cumsum(torch.cat((torch.tensor([0], device=my_device), original_batch[:-1])), dim=0)
        original_end_indices = original_start_indices + original_batch
        all_indices = torch.arange(len(original_tensor), device=my_device).repeat(original_batch.size(0), 1)
        original_mask = (all_indices >= original_start_indices.unsqueeze(1)) & (all_indices < original_end_indices.unsqueeze(1))
        grouped_tensor = torch.where(original_mask, original_tensor.unsqueeze(0), torch.tensor(-float('inf'), device=my_device))
        start_indices = torch.argmax(grouped_tensor, dim=1) - original_start_indices
        BFS_inputs = [(data['ligand'].mol[i], start_indices[i].item(), data['metal'].start_metal_index[i].item()) for i in range(len(start_indices))]
  
        BFS_results = []
        for bfs_input in BFS_inputs:
            result = BFS_search(*bfs_input) 
            BFS_results.append(result)
        data['ligand'].seq_order, data['ligand'].seq_parent = map(lambda x: torch.tensor(list(chain(*x))).to(my_device), zip(*BFS_results))

        if docking:
            # select the focal residue
            # graph norm through a interaction graph
            data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch = self.construct_interaction_graph(data, pro_nodes_num, train=False)
            pos, pred_xyz, true_xyz, sample_batch = self.sampling_atom_pos(data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=False)
            # # loss     
            # rmsd_losss = self.cal_rmsd(pos_ture=true_xyz, pos_pred=pred_xyz, batch=sample_batch, if_r=True)
            # print('rmsd_losses', rmsd_losss.mean().item())
        else:
            pos = xyz
        lig_pose = pos[pro_nodes_num:]
        # scoring
        if scoring:
            dist, dist_ca = self.scoring_layers.cal_pair_dist_and_select_nearest_residue_4vs(lig_pos=lig_pose, lig_batch=data['ligand'].batch, 
                                                              pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch, C_mask=C_mask, B=B, N_l=N_l)
            # scoring loss
            score = self.scoring_PL_conformation(pi, sigma, mu, dist, dist_threhold, c_batch, batch_size)
        else:
            score = torch.zeros(batch_size, 1)
        return lig_pose, score, pi, sigma, mu, c_batch, batch_size, C_mask, B, N_l
    
    

