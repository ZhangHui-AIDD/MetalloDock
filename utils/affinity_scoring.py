#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from prefetch_generator import BackgroundGenerator
import rmsd
import warnings
from tqdm import tqdm
# dir of current
warnings.filterwarnings('ignore')
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from utils.fns import Early_stopper, set_random_seed
from dataset.graph_obj import VSTestGraphDataset_Fly_SDFMOL2_Refined
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.MetalloDock import MetalloDock
from utils.post_processing import out_pose, correct_pos
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



argparser = argparse.ArgumentParser()
argparser.add_argument('--ligand_path', type=str,
                    default=f'./predicted_poses/docking_test/0',
                    # default=f'../dataset/metallo_vs/PA/result',
                    help='the ligand files path')
argparser.add_argument('--ligand_name', type=str,
                    default=f'7oyo_pred',
                    # default='',
                    help='the ligand file name')
argparser.add_argument('--pocket_file', type=str,
                    default=f'../dataset/metallo_complex/7oyo/7oyo_pro_pocket.pdb',
                    # default=f'../dataset/metallo_vs/PA/receptor_pocket.pdb',
                    help='the pocket files path')
argparser.add_argument('--voc_file', type=str,
                    default='../dataset/chembl_fragments_voc_dict.pkl',
                    help='the CHEMBL fragments voc dict')
argparser.add_argument('--model_file', type=str,
                    default='../trained_models/docking_metallodock_time.pkl',
                    help='model file')
argparser.add_argument('--mode', type=str,
                    default='docking',
                    help='Type of results to be scored: docking/vs')
argparser.add_argument('--suffix', type=str,
                    default='pred',
                    help='type of postprocessing that require scoring: pred(raw)/align/ff/uff')
argparser.add_argument('--out_dir', type=str,
                    default=f'./',
                    # default=f'../dataset/metallo_vs/PA/result',
                    help='dir for recording binding poses and binding scores')
argparser.add_argument('--batch_size', type=int,
                    default=64,
                    help='batch size')
argparser.add_argument('--random_seed', type=int,
                    default=2024,
                    help='random_seed')
args = argparser.parse_args()

def metallo_scoring(args):
    set_random_seed(args.random_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    # device
    device_id = 0
    if torch.cuda.is_available():
        my_device = f'cuda:{device_id}'
    else:
        my_device = 'cpu'
    # model
    model = MetalloDock()  
    model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
    model.to(my_device)
    stopper = Early_stopper(model_file=args.model_file, mode='lower', patience=10)
    stopper.load_model(model_obj=model, my_device=my_device, strict=False)
    # dataset
    if args.mode == 'vs':
        ligand_names_list = [i.split('.')[0] for i in os.listdir(args.ligand_path) if args.suffix in i]
    else:
        ligand_names_list = [args.ligand_name]
    test_dataset = VSTestGraphDataset_Fly_SDFMOL2_Refined(protein_file=args.pocket_file, ligand_path=args.ligand_path, ligand_files=ligand_names_list, voc_file=args.voc_file, my_device=my_device)
    model = torch.compile(model)
    # dataloader
    test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=16, prefetch_factor=4, follow_batch=[], pin_memory=True)  
    # time
    pdb_ids = []
    labels = []
    binding_scores = []
    binding_scores_ff_corrected = []
    binding_scores_align_corrected = []
    binding_scores_uff_corrected = []
    with torch.no_grad():
        model.eval()
        for idx, data in enumerate(tqdm(test_dataloader)):   
            try:
                # to device
                data = data.to(my_device)
                batch_size = data['ligand'].batch[-1] + 1
                # scoring
                mdn_score = model.module.scoring_PL_complex(data)
                pdb_ids.extend([pdbid.split('_')[0] for pdbid in data.pdb_id])
                binding_scores.extend(mdn_score.cpu().numpy().tolist())
                labels.extend([1 if i.startswith(('CHEMBL', 'BDB')) else 0 for i in data.pdb_id])
            except:
                print('ERROR!',data.pdb_id)
                continue
        # out to csv
        if args.mode == 'vs':
            df_score = pd.DataFrame(list(zip(pdb_ids, binding_scores, labels)), columns=['pdb_id', f'{args.suffix}_score', 'label'])
            df_score.to_csv(f'{args.out_dir}/score.csv', index=False)
        else:
            df_score = pd.DataFrame(list(zip(pdb_ids, binding_scores)), columns=['pdb_id', f'mdn_score'])
            df_score.to_csv(f'{args.out_dir}/affinity_score.csv', index=False)

if __name__ == '__main__':
    metallo_scoring(args)
