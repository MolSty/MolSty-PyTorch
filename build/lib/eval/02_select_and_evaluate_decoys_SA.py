#!/usr/bin/env python
"""
Usage:
    select_and_evaluate_decoys.py [options]

Options:
    -h --help                       Show this screen.
    --data_path NAME                Path to data file or directory containing multiple files
    --output_path NAME              Path to output location [default: ./]
    --dataset_name NAME             Name of dataset (Options: dude, dude-ext, dekois, MUV, ALL)
    --num_decoys_per_active INT     Number of decoys to select per active [default: 50]
    --min_num_candidates INT        Minimum number of candidate decoys [default: 100]
    --min_active_size INT           Minimum number of atoms in active molecule [default: 10]
    --num_cores INT                 Number of cores to use [default: 1]
    --max_idx_cmpd INT              Maximum number of decoys per active to consider [default: 10000]
"""

import os, csv

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import decoy_utils

import sascorer

from joblib import Parallel, delayed
from docopt import docopt
def get_sa(x):
    try:
        return sascorer.calculateScore(x)
    except:
        return -1

def get_mol_features(mol):
    try:
        fp3 = AllChem.GetMACCSKeysFingerprint(mol).ToBitString()
        fp3 = [int(item) for item in fp3]
        if np.sum(fp3) == 0:
            return fp3
        fp3 = (np.array(fp3)/(np.sum(fp3**2)**0.5))
        return list(fp3)
    except:
        return 0
# Worker function
def select_and_evaluate_decoys(f, target, idx, file_loc='./', output_loc='./', 
                               dataset='ALL', num_cand_dec_per_act=100, 
                               num_dec_per_act=50, max_idx_cmpd=10000):
    print("Processing: ", f)
    dec_results = [f]
    dec_results.append(dataset)
    # Read data
    data = decoy_utils.read_paired_file(file_loc+f)
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    tmp = [Chem.MolFromSmiles(d[0]) for d in data]
    data = [d for idx, d in enumerate(data) if tmp[idx] is not None \
            and tmp[idx].GetNumHeavyAtoms()>min_active_size]
    data = pd.DataFrame(data, columns=['act', 'dec'])
    data['dec_mol'] = data['dec'].apply(lambda x: Chem.MolFromSmiles(x))
    data['dec_SA'] = data['dec_mol'].apply(lambda x: get_sa(x))
    data = data.loc[data['dec_SA']!=-1].reset_index(drop=True)
    tmp_dic = {x:get_mol_features(Chem.MolFromSmiles(x)) for x in set(data['act'])}
    data['act_fp'] = data['act'].apply(lambda x: tmp_dic[x])
    data['dec_fp'] = data['dec_mol'].apply(lambda x: get_mol_features(x))
    data['similarity'] = data[['act_fp', 'dec_mol']].values.tolist()
    data['similarity'] = data['similarity'].apply(
        lambda x: np.sum(np.array(get_mol_features(x[0])) * 
                         np.array(get_mol_features(x[1]))))
    data['score'] = data['similarity']
    result = []
    for key, tmp_df in data.groupby('act'):
        tmp_df = tmp_df.sort_values('score', ascending=False)
        tmp_df = tmp_df.reset_index(drop=True)
        for i in range(min([30, tmp_df.shape[0]])):
            result.append([key, tmp_df['dec'].values[i]])
    result = pd.DataFrame(result, columns=['act', 'dec'])
    output_name = output_loc + \
        f'/{target}_{idx}_selected_{num_dec_per_act}_{num_cand_dec_per_act}.smi'
    
    result.to_csv(output_name, index=False, header=None, sep=' ')
    
    decoy_smis_gen = list(set(result['dec']))
    decoy_mols_gen = [Chem.MolFromSmiles(smi) for smi in decoy_smis_gen]
    active_smis_gen = list(set(result['act']))
    active_mols_gen = [Chem.MolFromSmiles(smi) for smi in active_smis_gen]
    dataset = 'dude'
    print('Calc props for chosen decoys')
    actives_feat = decoy_utils.calc_dataset_props_dude(active_mols_gen)
    decoys_feat = decoy_utils.calc_dataset_props_dude(decoy_mols_gen)

    print('ML model performance')
    print(actives_feat.shape)
    print(decoys_feat.shape)
    dec_results.extend(list(decoy_utils.calc_xval_performance(
        actives_feat, decoys_feat, n_jobs=1)))

    print('DEKOIS paper metrics (LADS, DOE, Doppelganger score)')
    dec_results.append(decoy_utils.doe_score(actives_feat, decoys_feat))
    lads_scores = decoy_utils.lads_score_v2(active_mols_gen, decoy_mols_gen)
    dec_results.append(np.mean(lads_scores))
    dg_scores, dg_ids = decoy_utils.dg_score(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(dg_scores), max(dg_scores)])
    
    print('Save decoy mols')
    print(dec_results)
    return dec_results

import argparse
if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='tmp.csv')
    parser.add_argument('--output_path', default='./eval/results/')
    parser.add_argument('--dataset_name', default='dude')
    parser.add_argument('--num_decoys_per_active', default=30)
    parser.add_argument('--min_num_candidates', default=500)
    parser.add_argument('--min_active_size', default=10)
    parser.add_argument('--max_idx_cmpd', default=10000)
    parser.add_argument('--target')
    parser.add_argument('--idx')
    # parser.add_argument('--times')
    args = parser.parse_args()
    
    target = args.target
    idx = args.idx
    file_loc = args.data_path
    file_loc = f'{target}_{idx}_{file_loc}'
    # times = int(args.times)
    output_loc = args.output_path
    dataset = args.dataset_name
    # num_dec_per_act = args.num_decoys_per_active * times
    num_cand_dec_per_act = args.min_num_candidates #* times
# =============================================================================
#     if num_cand_dec_per_act == -1:
#         with open('./num_cand_dec_per_act.txt', 'r') as f:
#             num_cand_dec_per_act = ''.join(f.readline())
#             num_cand_dec_per_act = int(num_cand_dec_per_act.strip('/n'))
#             print(num_cand_dec_per_act)
# =============================================================================
    num_dec_per_act = args.num_decoys_per_active # * times
    max_idx_cmpd = int(args.max_idx_cmpd)
    min_active_size = int(args.min_active_size)

    # Declare metric variables
    columns = ['File name', 'Dataset',
               'Orig num actives', 'Num actives', 'Num generated mols', 'Num unique gen mols',
               'AUC ROC - 1NN', 'AUC ROC - RF',
               'DOE score',
               'LADS score',
               'Doppelganger score mean', 'Doppelganger score max',
               ]

    # Populate CSV file with headers
    with open(output_loc+f'/{target}_results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)

    # Select decoys and evaluate
# =============================================================================
#     with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
#         results = parallel(delayed(select_and_evaluate_decoys)(
#             f,file_loc=file_loc, output_loc=output_loc, dataset=dataset, 
#             num_cand_dec_per_act=num_cand_dec_per_act, 
#             num_dec_per_act=num_dec_per_act, 
#             max_idx_cmpd=max_idx_cmpd) for f in res_files)
# =============================================================================
    from select_new import run
    with open(output_loc+f'/{target}_results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        print('*'*100)
        results = select_and_evaluate_decoys(
                f=file_loc, target=target, idx=idx, 
                file_loc=output_loc, 
                output_loc=output_loc, dataset=dataset, 
                num_cand_dec_per_act=num_cand_dec_per_act, num_dec_per_act=num_dec_per_act, 
                max_idx_cmpd=max_idx_cmpd)
        writer.writerow(results) 
        
        output_name = f'{target}_{idx}_selected_{num_dec_per_act}_{num_cand_dec_per_act}_output.smi'
        run(f'./eval/results/{target}_{idx}_selected_{num_dec_per_act}_{num_cand_dec_per_act}.smi', 
            './eval/results/'+output_name)
        
# =============================================================================
#         print('='*100)
#         results = select_and_evaluate_decoys(
#                 f=output_name, target=target, idx=idx, 
#                 file_loc=output_loc, 
#                 output_loc=output_loc, dataset=dataset, 
#                 num_cand_dec_per_act=1,#*times, 
#                 num_dec_per_act=1,#*times, 
#                 max_idx_cmpd=max_idx_cmpd)
#         writer.writerow(results) 
# =============================================================================
