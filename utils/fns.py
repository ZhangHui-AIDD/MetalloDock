#!usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import random
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from joblib import load, dump
from torch.distributions import Normal
from collections import Counter
from scipy.signal import find_peaks
from scipy.stats import norm
from queue import Queue


def BFS_search(mol, start_atom_idx=None, start_metal_idx=9999, random_seed=None):
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
    parent_lis = [start_metal_idx]
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


def mdn_loss_fn(pi, sigma, mu, y):
    epsilon = 1e-8
    normal = Normal(mu, sigma+epsilon)
    loglik = normal.log_prob(y.expand_as(normal.loc)) + epsilon
    pi = torch.softmax(pi, dim=1)
    loss = -torch.logsumexp(torch.log(pi) + loglik, dim=1)
    return loss

def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(1)        
    return prob


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def partition_job(data_lis, job_n, total_job=4, strict=False):
    length = len(data_lis)
    step = length // total_job
    if not strict:
        if job_n == total_job - 1:
            return data_lis[job_n * step:]
    return data_lis[job_n * step: (job_n + 1) * step]


def save_graph(dst_file, data):
    dump(data, dst_file)


def load_graph(src_file):
    return load(src_file)


def partition_job(data_lis, job_n, total_job=4, strict=False):
    length = len(data_lis)
    step = length // total_job
    if length % total_job == 0:
        return data_lis[job_n * step: (job_n + 1) * step]
    else:
        if not strict:
            if job_n == total_job - 1:
                return data_lis[job_n * step:]
            else:
                return data_lis[job_n * step: (job_n + 1) * step]
        else:
            step += 1
            if job_n * step <= length-1:
                data_lis += data_lis
                return data_lis[job_n * step: (job_n + 1) * step]
            else:
                return random.sample(data_lis, step)


def read_equibind_split(split_file):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Early_stopper(object):
    def __init__(self, model_file, mode='higher', patience=70, tolerance=0.0):
        self.model_file = model_file
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def load_model(self, model_obj, my_device, strict=False):
        '''Load model saved with early stopping.'''
        model_obj.load_state_dict(torch.load(self.model_file, map_location=my_device)['model_state_dict'], strict=strict)

    def save_model(self, model_obj):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model_obj.state_dict()}, self.model_file)

    def step(self, score, model_obj):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model_obj)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_model(model_obj)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'# EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        print(f'# Current best performance {float(self.best_score):.3f}')
        return self.early_stop

