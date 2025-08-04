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
from dataset.graph_obj import PDBBindGraphDataset
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
argparser.add_argument('--graph_file_dir', type=str,
                       default = '../dataset/test_graph', 
                       help='the graph files path')
argparser.add_argument('--model_file', type=str,
                       default='../trained_models/docking_metallodock_time.pkl',
                       help='model file')
argparser.add_argument('--frag_voc_file', type=str, 
                       default='../dataset/chembl_fragments_voc_dict.pkl')
argparser.add_argument('--out_dir', type=str,
                       default='./predicted_poses/docking_test',
                       help='dir for recording binding poses and binding scores')
argparser.add_argument('--protein_path', type=str,
                       default='../dataset/metallo_complex',
                       help='If the input is a file path, the script will read it directly; if the input is a folder path, it will assemble the file path using the pdbid.')
argparser.add_argument('--docking', type=bool,
                       default=True,
                       help='whether generating binding poses')
argparser.add_argument('--scoring', type=bool,
                       default=True,
                       help='whether predict binding affinities')
argparser.add_argument('--refine', type=bool,
                       default=True,
                       help='whether use ff to refine')
argparser.add_argument('--min_method', type=bool,
                       default='openff2',
                       help='uff for faster energy minimization in VS; openff2 to obtain more accurate binding conformations in docking.')
argparser.add_argument('--batch_size', type=int,
                       default=64,
                       help='batch size')
argparser.add_argument('--random_seed', type=int,
                       default=2024,
                       help='random_seed')
argparser.add_argument('--csv_file', type=str,
                       default='../dataset/test_pdbid.csv',   
                       help='the csv file with dataset split')
args = argparser.parse_args()
set_random_seed(args.random_seed)

df = pd.read_csv(args.csv_file)
test_pdb_ids = df[df.loc[:, 'time_split'] == 'test'].loc[:, 'pdb_id'].values.tolist()
# dataset
test_dataset = PDBBindGraphDataset(src_dir='',
                                   dst_dir=args.graph_file_dir,
                                   pdb_ids=test_pdb_ids,
                                   voc_file=args.frag_voc_file,
                                   n_job=1)


# dataloader
test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, follow_batch=['atom2res'], pin_memory=True)
# device
device_id = 0
if torch.cuda.is_available():
    my_device = f'cuda:{device_id}'
else:
    my_device = 'cpu'
# model
model = MetalloDock(hierarchical=True)
model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
model.to(my_device)
# stoper
stopper = Early_stopper(model_file=args.model_file,
                        mode='lower', patience=10)
# load existing model
stopper.load_model(model_obj=model, my_device=my_device, strict=False)
for re in range(1):
    model.eval()
    out_dir = f'{args.out_dir}/{re}'
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            print(f"Ligands to be docked: {data.pdb_id}")
            try:
                # to device
                data = data.to(my_device)
                pos_pred, mdn_score, _, _, _, _, _, _, _, _ = model.module.ligand_docking_coordination(data, docking=args.docking, scoring=args.scoring, dist_threhold=5)
                data.pos_preds = pos_pred
                out_pose(data, out_dir=out_dir, out_init=True)

                if args.refine:
                    print("Post-processing of predicted conformations")
                    try:
                        correct_poses = correct_pos(data, out_dir=out_dir, protein_path=args.protein_path, min_method=args.min_method, out_corrected=True)   
                    except:
                        continue 
            except Exception as e:
                print(f'Error in {data.pdb_id}: {e}')
        print(f'Docking results have been saved to the {out_dir}.')






