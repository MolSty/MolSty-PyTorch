import os, csv
import pandas as pd
import numpy as np
from select_new import run
from rdkit import Chem

import decoy_utils

def select_and_evaluate_decoys(f, file_loc='./', dataset='dude', num_cand_dec_per_act=100, num_dec_per_act=50, max_idx_cmpd=10000):
    print("Processing: ", f)
    dec_results = [f]
    dec_results.append(dataset)
    # Read data
    data = decoy_utils.read_paired_file(file_loc+f)
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    seen = set()
    data = [d for d in data if Chem.MolFromSmiles(d[0]) is not None and Chem.MolFromSmiles(d[0]).GetNumHeavyAtoms()>min_active_size]
    unique_data = [x for x in data if not (tuple(x) in seen or seen.add(tuple(x)))]
    
    in_smis = [d[0] for d in data]
    gen_smis = [d[1] for d in data]
    dec_results.extend([len(set(in_smis)), len(data), len(unique_data)])

    print('Calculate properties of in_smis and gen_mols')
    used = set([])
    in_smis_set = [x for x in in_smis if x not in used and (used.add(x) or True)]

    active_mols_gen = [Chem.MolFromSmiles(smi) for smi in in_smis_set]
    decoy_mols_gen = [Chem.MolFromSmiles(smi) for smi in gen_smis]
    dataset = 'dude'
    print('Calc props for chosen decoys')
    actives_feat = decoy_utils.calc_dataset_props_dude(active_mols_gen)
    decoys_feat = decoy_utils.calc_dataset_props_dude(decoy_mols_gen)

    print('ML model performance')
    dec_results.extend(list(decoy_utils.calc_xval_performance(actives_feat, decoys_feat, n_jobs=1)))

    print('DEKOIS paper metrics (LADS, DOE, Doppelganger score)')
    dec_results.append(decoy_utils.doe_score(actives_feat, decoys_feat))
    lads_scores = decoy_utils.lads_score_v2(active_mols_gen, decoy_mols_gen)
    dec_results.append(np.mean(lads_scores))
    dg_scores, dg_ids = decoy_utils.dg_score(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(dg_scores), max(dg_scores)])
    
    print(dec_results)
    return dec_results

root = '../data/DeepCoy_decoys/DeepCoy-DUDE-SMILES/'
target_lst = os.listdir(root)

max_idx_cmpd = 10000
min_active_size = 10# int(args.min_active_size)
dataset = 'dude'
with open('./eval/results_deepcoy.csv', 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for target in target_lst:
        N = 50
        num_cand_dec_per_act = N
        num_dec_per_act = N
    
        results = select_and_evaluate_decoys(
                f=target, 
                file_loc=root, 
                dataset=dataset, 
                num_cand_dec_per_act=num_cand_dec_per_act, num_dec_per_act=num_dec_per_act, 
                max_idx_cmpd=max_idx_cmpd)
        writer.writerow(results)