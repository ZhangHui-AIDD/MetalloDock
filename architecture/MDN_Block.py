#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class MDN_Block(nn.Module):
    def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, 
                 dist_threhold=1000):
        super(MDN_Block, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate)) 
        self.start_donar_selector =  nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate), nn.Linear(hidden_dim, 1), nn.Sigmoid()) 
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(hidden_dim, 18)
        self.bond_types = nn.Linear(hidden_dim*2, 5)        
        self.dist_threhold = dist_threhold
    
    def forward(self, lig_s, lig_batch, pro_s, pro_batch, donar_metal_mask, edge_index):
        # to batch
        h_l_x, l_mask = to_dense_batch(lig_s, lig_batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)

        assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]

        C = torch.cat((h_l_x, h_t_x), -1)
        # de-batch
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        C = C[C_mask]

        # metal_donar_mask
        donar_metal = C[donar_metal_mask]
        #predict donar
        coordinated_donar = self.start_donar_selector(donar_metal).squeeze(-1)
        # predict interaction
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(lig_s.device)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        atom_types = self.atom_types(lig_s)
        bond_types = self.bond_types(torch.cat([lig_s[edge_index[0]],lig_s[edge_index[1]]], axis=1))
        
        return pi, sigma, mu, C_batch, atom_types, bond_types, coordinated_donar, C_mask, B, N_l 

    def cal_pair_dist(self, lig_pos, pro_pos, pro_batch, lig_batch):
        h_l_pos, _ = to_dense_batch(lig_pos, lig_batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3))[self.C_mask]
        return dist.unsqueeze(1).detach()

    def cal_pair_dist_ca(self, lig_pos, pro_pos, pro_batch, lig_batch):
        h_l_pos, _ = to_dense_batch(lig_pos, lig_batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        dist = self.compute_euclidean_distances_matrix_ca(h_l_pos, h_t_pos)[self.C_mask]
        return dist.unsqueeze(1).detach()

    def cal_pair_dist_and_select_nearest_residue(self, lig_pos, pro_pos, pro_batch, lig_batch):
        h_l_pos, _ = to_dense_batch(lig_pos, lig_batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3)).detach()
        ca_dis_matrix = self.compute_euclidean_distances_matrix_ca(h_l_pos, h_t_pos[:, :, 1, ...]).detach()
        return dist[self.C_mask].unsqueeze(-1), ca_dis_matrix[self.C_mask].unsqueeze(-1)

    
    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        # return dists**0.5
        return torch.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,24),10000).min(axis=-1)[0]
    
    def compute_euclidean_distances_matrix_ca(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        # return dists**0.5
        return torch.nan_to_num((dists**0.5),10000)

    def cal_pair_dist_and_select_nearest_residue_4vs(self, lig_pos, pro_pos, pro_batch, lig_batch, C_mask, B, N_l):
        h_l_pos, _ = to_dense_batch(lig_pos, lig_batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        dist = self.compute_euclidean_distances_matrix_4vs(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3), B, N_l).detach()
        ca_dis_matrix = self.compute_euclidean_distances_matrix_ca(h_l_pos, h_t_pos[:, :, 1, ...]).detach()
        return dist[C_mask].unsqueeze(-1), ca_dis_matrix[C_mask].unsqueeze(-1)

    
    def compute_euclidean_distances_matrix_4vs(self, X, Y, B, N_l):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        # return dists**0.5
        return torch.nan_to_num((dists**0.5).view(B, N_l,-1,24),10000).min(axis=-1)[0]

    

