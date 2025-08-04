#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
import time
import copy
import torch.nn as nn
import pandas as pd
import torch.optim
import numpy as np
from torch_geometric.loader import DataLoader
from prefetch_generator import BackgroundGenerator
import rmsd
from tqdm import tqdm
# dir of current
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from utils.fns import Early_stopper, set_random_seed, BFS_search
from dataset.graph_obj import VSTestGraphDataset_Fly_SDFMOL2_Refined
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.MetalloDock import MetalloDock
from utils.post_processing import out_pose, correct_pos

metallo_ascii = r"""
  ____  _     _ _   _ _   _ ____  _     _____ ____  _   _ ____  _     _     
        __  __      _        _ _       ____             _    
        |  \/  | ___| |_ __ _| | | ___ |  _ \  ___   ___| | __
        | |\/| |/ _ \ __/ _` | | |/ _ \| | | |/ _ \ / __| |/ /
        | |  | |  __/ || (_| | | | (_) | |_| | (_) | (__|   < 
        |_|  |_|\___|\__\__,_|_|_|\___/|____/ \___/ \___|_|\_\
  ____  _     _ _   _ _   _ ____  _     _____ ____  _   _ ____  _     _     
"""
print(metallo_ascii)


class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--ligand_path', type=str,
                    default=f'../dataset/metallo_complex/7oyo',
                    help='the ligand files path')
argparser.add_argument('--ligand_name', type=str,
                    default=f'7oyo_ligand',
                    help='the ligand file name')
argparser.add_argument('--pocket_file', type=str,
                    default=f'../dataset/metallo_complex/7oyo/7oyo_pro_pocket.pdb',
                    help='the pocket files path')
argparser.add_argument('--coordination_idx', type=int,
                    default=12,
                    help='the specified ligand coordination atom indices, used as the starting point for autoregressive generation')                   
argparser.add_argument('--voc_file', type=str,
                    default='../dataset/chembl_fragments_voc_dict.pkl',
                    help='the CHEMBL fragments voc dict')
argparser.add_argument('--model_file', type=str,
                    default='../trained_models/docking_metallodock_time.pkl',
                    help='model file')
argparser.add_argument('--refine', type=bool,
                    default=True,
                    help='whether use ff to refine')
argparser.add_argument('--min_method', type=str,
                    default='openff2',
                    help='uff for faster energy minimization in VS; openff2 to obtain more accurate binding conformations in docking.')
argparser.add_argument('--docking', type=bool,
                    default=True,
                    help='whether generating binding poses')
argparser.add_argument('--scoring', type=bool,
                    default=True,
                    help='whether predict binding affinities')
argparser.add_argument('--out_dir', type=str,
                    default=f'./predicted_poses/docking_custom',
                    help='dir for recording binding poses and binding scores')
argparser.add_argument('--batch_size', type=int,
                    default=1,
                    help='batch size')
argparser.add_argument('--random_seed', type=int,
                    default=2024,
                    help='random_seed')
args = argparser.parse_args()
set_random_seed(args.random_seed)

# device
device_id = 0
if torch.cuda.is_available():
    my_device = f'cuda:{device_id}'
else:
    my_device = 'cpu'
# dataset
ligand_names_list = [args.ligand_name]
test_dataset = VSTestGraphDataset_Fly_SDFMOL2_Refined(protein_file=args.pocket_file, ligand_path=args.ligand_path, 
                                                      ligand_files=ligand_names_list, voc_file=args.voc_file, my_device=my_device)
# dataloader
test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, follow_batch=['atom2res'], pin_memory=True)
# model
model = MetalloDock(hierarchical=True)
model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
model.to(my_device)
stopper = Early_stopper(model_file=args.model_file, mode='lower', patience=10)
stopper.load_model(model_obj=model, my_device=my_device, strict=False)
for re in range(1):
    model.eval()
    out_dir = f'{args.out_dir}/{re}'
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            print(f"Ligands to be docked: {data.pdb_id}, with coordination specified at ligand atom number {args.coordination_idx}.")
            try:
                # to device
                data = data.to(my_device)
                seq_order, seq_parent = BFS_search(data['ligand'].mol[0], start_atom_idx=args.coordination_idx, 
                                                   start_metal_idx=data['metal'].start_metal_index)
                seq_parent[0] = data['metal'].start_metal_index
                data['ligand'].seq_order = torch.tensor(seq_order[:], dtype=torch.long).to(my_device)
                data['ligand'].seq_parent = torch.tensor(seq_parent[:], dtype=torch.long).to(my_device)

                pos_pred, mdn_score = model.module.ligand_docking(data, docking=args.docking, scoring=args.scoring, dist_threhold=5)
                data.pos_preds = pos_pred
                out_pose(data, out_dir=out_dir, out_init=True)

                if args.refine:
                    print("Post-processing of predicted conformations")
                    try:
                        correct_poses = correct_pos(data, out_dir=out_dir, protein_path=args.pocket_file, min_method=args.min_method, out_corrected=True)   
                    except:
                        continue 
            except Exception as e:
                print(f'Error in {data.pdb_id}: {e}')
        print(f'Docking results have been saved to the {out_dir}.')






