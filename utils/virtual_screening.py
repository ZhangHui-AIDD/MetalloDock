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
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from utils.fns import Early_stopper, set_random_seed
from dataset.graph_obj import VSTestGraphDataset_Fly_SDFMOL2_Refined, get_mol2_xyz_from_cmd
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.MetalloDock import MetalloDock
from utils.post_processing import out_pose, correct_pos
import torch.multiprocessing as mp

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


pre_parser = argparse.ArgumentParser()
pre_parser.add_argument('--target', type=str, default='PA', help='the VS target')
pre_args, remaining_args = pre_parser.parse_known_args()
target = pre_args.target

argparser = argparse.ArgumentParser()
argparser.add_argument('--ligand_file', type=str,
                    default=f'../dataset/metallo_vs/{target}/ligands',
                    help='the ligand files path')
argparser.add_argument('--pocket_file', type=str,
                    default=f'../dataset/metallo_vs/{target}/receptor_pocket.pdb',
                    help='the pocket files path')
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
                    default='uff',
                    help='uff for faster energy minimization in VS; openff2 to obtain more accurate binding conformations in docking.')
argparser.add_argument('--out_dir', type=str,
                    default=f'../dataset/metallo_vs/{target}/result',
                    help='dir for recording binding poses and binding scores')
argparser.add_argument('--batch_size', type=int,
                    default=64,
                    help='batch size')
argparser.add_argument('--random_seed', type=int,
                    default=2024,
                    help='random_seed')
args = argparser.parse_args(remaining_args)

def MetalloDock_VS_pipeline(args):
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
    # stoper
    stopper = Early_stopper(model_file=args.model_file,
                            mode='lower', patience=10)
    model_name = args.model_file.split('/')[-1].split('_')[1]
    print('# load model')
    # load existing model
    stopper.load_model(model_obj=model, my_device=my_device, strict=False)
    # dataset
    ligand_names_list = [i.split('.')[0] for i in os.listdir(args.ligand_file)]
    print('VS_dataset_num:',len(ligand_names_list))
    test_dataset = VSTestGraphDataset_Fly_SDFMOL2_Refined(protein_file=args.pocket_file, ligand_path=args.ligand_file, ligand_files=ligand_names_list, voc_file=args.voc_file, my_device=my_device)
    model = torch.compile(model)
    # dataloader
    test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, prefetch_factor=4, follow_batch=[], pin_memory=True)  
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
                    print("Ligands to be virtually screened:", data.pdb_id)
                    start_time = time.perf_counter()
                    data = data.to(my_device)
                    batch_size = data['ligand'].batch[-1] + 1
                    # forward
                    lig_pose, mdn_score_pred, pi, sigma, mu, c_batch, batch_size, C_mask, B, N_l = model.module.ligand_docking_coordination(data)
                    data.pos_preds = lig_pose
                    out_pose(data, out_dir=args.out_dir, out_init=False)
                    if args.refine: 
                        try:
                            correct_poses = correct_pos(data, out_dir=args.out_dir, protein_path=args.pocket_file, min_method=args.min_method, out_corrected=True)   
                        except:
                            continue 
                        align_corrected_pos = torch.from_numpy(np.concatenate([i[0] for i in correct_poses], axis=0)).to(my_device)
                        ff_corrected_pos = torch.from_numpy(np.concatenate([i[1] for i in correct_poses], axis=0)).to(my_device)
                        min_corrected_pos = torch.from_numpy(np.concatenate([i[2] for i in correct_poses], axis=0)).to(my_device)

                        dist_ff, dist_ca_ff = model.module.scoring_layers.cal_pair_dist_and_select_nearest_residue_4vs(lig_pos=ff_corrected_pos, lig_batch=data['ligand'].batch, 
                                                                            pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch, C_mask=C_mask, B=B, N_l=N_l)
                        mdn_score_pred_ff_corrected = model.module.scoring_PL_conformation(pi, sigma, mu, dist_ff, 5.0, c_batch, batch_size)
                        dist_align, dist_ca_align = model.module.scoring_layers.cal_pair_dist_and_select_nearest_residue_4vs(lig_pos=align_corrected_pos, lig_batch=data['ligand'].batch, 
                                                                            pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch, C_mask=C_mask, B=B, N_l=N_l)
                        mdn_score_pred_align_corrected = model.module.scoring_PL_conformation(pi, sigma, mu, dist_align, 5.0, c_batch, batch_size)
                        dist_min, dist_ca_min = model.module.scoring_layers.cal_pair_dist_and_select_nearest_residue_4vs(lig_pos=min_corrected_pos, lig_batch=data['ligand'].batch, 
                                                    pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch, C_mask=C_mask, B=B, N_l=N_l)
                        mdn_score_pred_min_corrected = model.module.scoring_PL_conformation(pi, sigma, mu, dist_min, 5.0, c_batch, batch_size)

                        binding_scores_align_corrected.extend(mdn_score_pred_align_corrected.cpu().numpy().tolist())
                        binding_scores_ff_corrected.extend(mdn_score_pred_ff_corrected.cpu().numpy().tolist())
                        binding_scores_uff_corrected.extend(mdn_score_pred_min_corrected.cpu().numpy().tolist())
                    pdb_ids.extend(data.pdb_id)
                    binding_scores.extend(mdn_score_pred.cpu().numpy().tolist())
                    labels.extend([1 if i.startswith(('CHEMBL', 'BDB')) else 0 for i in data.pdb_id])
                    end_time = time.perf_counter()
                    print('MetalloDock_VS cost time',start_time-end_time)
                except:
                    print('Error in VS:',data.pdb_id)
                    continue
        # out to csv
        df_score = pd.DataFrame(list(zip(pdb_ids, binding_scores, labels)), columns=['pdb_id', 'score', 'label'])
        if args.refine: 
            df_score['ff_score'] = binding_scores_ff_corrected
            df_score['aligned_score'] = binding_scores_align_corrected
            df_score['min_score'] = binding_scores_uff_corrected
        df_score.to_csv(f'{args.out_dir}/score.csv', index=False)

if __name__ == '__main__':
    MetalloDock_VS_pipeline(args)
