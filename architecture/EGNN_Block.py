#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pandas as pd
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter, scatter_mean

class EGNN(nn.Module):
    def __init__(self, dim_in, dim_tmp, edge_in, edge_out, num_head=8, drop_rate=0.15, update_pos=False):
        super().__init__()
        assert dim_tmp % num_head == 0
        self.edge_dim = edge_in
        self.num_head = num_head # 4
        self.dh = dim_tmp // num_head # 32
        self.dim_tmp = dim_tmp # 128
        self.q_layer = nn.Linear(dim_in, dim_tmp)
        self.k_layer = nn.Linear(dim_in, dim_tmp)
        self.v_layer = nn.Linear(dim_in, dim_tmp)
        self.m_layer = nn.Sequential(
            nn.Linear(edge_in+1, dim_tmp),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(), 
            nn.Linear(dim_tmp, dim_tmp)
            )
        self.m2f_layer = nn.Sequential(
            nn.Linear(dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.e_layer = nn.Sequential(
            nn.Linear(dim_tmp, edge_out),
            nn.Dropout(p=drop_rate))
        self.gate_layer = nn.Sequential(
            nn.Linear(3*dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.layer_norm_1 = GraphNorm(dim_tmp)
        self.layer_norm_2 = GraphNorm(dim_tmp)
        self.fin_layer = nn.Sequential(
            nn.Linear(dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_tmp, dim_tmp)
            )
        self.update_pos = update_pos
        if update_pos:
            self.update_layer = coords_update(dim_dh=self.dh, num_head=num_head, drop_rate=drop_rate)
    
    def forward(self, node_s, edge_s, edge_index, generate_node_dist, pos, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num, batch):
        q_ = self.q_layer(node_s)
        k_ = self.k_layer(node_s)
        v_ = self.v_layer(node_s)
        # message passing
        ## cal distance
        d_ij = torch.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(dim=-1)*0.1
        ## mask unkonwn distance    
        d_ij[mask_edge_inv] = -1
        ## cal attention
        m_ij = torch.cat([edge_s, d_ij], dim=-1)
        m_ij = self.m_layer(m_ij)
        k_ij = k_[edge_index[1]] * m_ij
        a_ij = ((q_[edge_index[0]] * k_ij)/math.sqrt(self.dh)).view((-1, self.num_head, self.dh))
        # Zip the lists and the a_ij variable together
        w_ij = softmax(torch.norm(a_ij, p=1, dim=2), index=edge_index[0]).unsqueeze(dim=-1)
        # update node and edge embeddings 
        node_s_new = self.m2f_layer(scatter(w_ij*v_[edge_index[1]].view((-1, self.num_head, self.dh)), index=edge_index[0], reduce='sum', dim=0).view((-1, self.dim_tmp)))
        edge_s_new = self.e_layer(a_ij.view((-1, self.dim_tmp)))
        g = torch.sigmoid(self.gate_layer(torch.cat([node_s_new, node_s, node_s_new-node_s], dim=-1)))
        node_s_new = self.layer_norm_1(g*node_s_new+node_s, batch)
        node_s_new = self.layer_norm_2(g*self.fin_layer(node_s_new)+node_s_new, batch)
        # update coords
        if self.update_pos:
            pos = self.update_layer(a_ij, pos, generate_node_dist, edge_index, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num)

        return node_s_new, edge_s_new, pos


class coords_update(nn.Module):
    def __init__(self, dim_dh, num_head, drop_rate=0.15):
        super().__init__()
        # self.correct_clash_num = correct_clash_num
        self.num_head = num_head
        self.attention2deltax = nn.Sequential(
            nn.Linear(dim_dh, dim_dh//2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_dh//2, 1)
        )
        self.weighted_head_layer = nn.Linear(num_head, 1, bias=False)

    def forward(self, a_ij, pos, generate_node_dist, edge_index, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num):
        edge_index_mask = edge_index[0] >= pro_nodes_num
        mask_edge_inv = mask_edge_inv.squeeze(dim=-1)[edge_index_mask]
        edge_index = edge_index[:, edge_index_mask]
        a_ij = a_ij[edge_index_mask, :, :]
        # cal vector delta_x
        delta_x = pos[edge_index[0]] - pos[edge_index[1]]
        delta_x = delta_x/(torch.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6 )
        # mask distance calculated by node with unknown coords
        delta_x[mask_edge_inv] = torch.zeros_like(delta_x[mask_edge_inv])
        # cal attention
        attention= self.weighted_head_layer(self.attention2deltax(a_ij).squeeze(dim=2))
        delta_x = delta_x*attention
        delta_x = scatter(delta_x, index=edge_index[0], reduce='sum', dim=0, dim_size=pos.size(0))
        # parent_node_idxes
        delta_x = delta_x[generate_node_idxes, :]
        return pos[generate_node_idxes, : ] + delta_x



