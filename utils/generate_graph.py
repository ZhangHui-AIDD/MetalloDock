#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import sys
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dataset import graph_obj       

argparser = argparse.ArgumentParser()
argparser.add_argument('--complex_path', type=str, default='../dataset/metallo_complex')               
argparser.add_argument('--frag_voc_file', type=str, default='../dataset/chembl_fragments_voc_dict.pkl')
argparser.add_argument('--job_n', type=int, default=0)
argparser.add_argument('--graph_path', type=str, default='../dataset/test_graph')                          
argparser.add_argument('--csv_file', type=str, default='../dataset/test_pdbid.csv')          
args = argparser.parse_args()
# init
complex_path = args.complex_path
df = pd.read_csv(args.csv_file)
pdb_ids = df['pdb_id'].values.tolist()                               
graph_path = args.graph_path
os.makedirs(graph_path, exist_ok=True)

test_dataset = graph_obj.PDBBindGraphDataset(src_dir=complex_path,
                                        dst_dir=graph_path,
                                        pdb_ids=pdb_ids,
                                        n_job=240,
                                        voc_file=args.frag_voc_file,
                                        verbose=True)
