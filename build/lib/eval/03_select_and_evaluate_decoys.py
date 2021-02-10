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

# Worker function
def select_and_evaluate_decoys(f, target, idx, file_loc='./', output_loc='./', 
                               dataset='ALL', num_cand_dec_per_act=100, 
                               num_dec_per_act=50, max_idx_cmpd=10000):
    print("Processing: ", f)
    dec_results = [f]
    dec_results.append(dataset)
    # Read data
    data = decoy_utils.read_paired_file(file_loc+f)
# =============================================================================
#         
# =============================================================================
    data = [d+[Chem.MolFromSmiles(d[1])] for d in data]
    lads_scores = decoy_utils.lads_score_v2(
        [Chem.MolFromSmiles(smi) for smi in list(set([d[0] for d in data]))], 
        [d[2] for d in data])
    data = [d for idx, d in enumerate(data) if lads_scores[idx]<0.5]
# =============================================================================
#     data = [d for d in data if AllChem.EmbedMolecule(
#         Chem.AddHs(d[2]), randomSeed=42) != -1]
# =============================================================================
    data = [d[:2] for d in data]
# =============================================================================
#         
# =============================================================================
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    seen = set()
    tmp = [Chem.MolFromSmiles(d[0]) for d in data]
    data = [d for idx, d in enumerate(data) if tmp[idx] is not None \
            and tmp[idx].GetNumHeavyAtoms()>min_active_size]
    unique_data = [x for x in data if not (tuple(x) in seen or seen.add(tuple(x)))]
    
    in_smis = [d[0] for d in data]
    in_mols = [Chem.MolFromSmiles(smi) for smi in in_smis]
    set_in_smis = list(set(in_smis))
    set_in_mols = [Chem.MolFromSmiles(smi) for smi in set_in_smis]
    gen_smis = [d[1] for d in data]
    gen_mols = [Chem.MolFromSmiles(smi) for smi in gen_smis]
    dec_results.extend([len(set(in_smis)), len(data), len(unique_data)])

    print('Calculate properties of in_smis and gen_mols')
    used = set([])
    in_smis_set = [x for x in in_smis if x not in used and (used.add(x) or True)]
    in_mols_set = [Chem.MolFromSmiles(smi) for smi in in_smis_set]
    if dataset == "dude_ext":
        in_props_temp = decoy_utils.calc_dataset_props_dude_extended(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props_dude_extended(gen_mols, verbose=True)
    elif dataset == "dekois":
        in_props_temp = decoy_utils.calc_dataset_props_dekois(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props_dekois(gen_mols, verbose=True)
    elif dataset == "MUV":
        in_props_temp = decoy_utils.calc_dataset_props_muv(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props_muv(gen_mols, verbose=True)
    elif dataset == "ALL":
        in_props_temp = decoy_utils.calc_dataset_props_all(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props_all(gen_mols, verbose=True)
    elif dataset == "dude":
        in_props_temp = decoy_utils.calc_dataset_props_dude(in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props_dude(gen_mols, verbose=True)
    else:
        print("Incorrect dataset")
        exit()
    in_mols_temp = list(in_smis_set) # copy
    in_props = []
    for i, smi in enumerate(in_smis):
        in_props.append(in_props_temp[in_mols_temp.index(smi)])

    in_basic_temp = decoy_utils.calc_dataset_props_basic(in_mols_set, verbose=True)
    in_mols_temp = list(in_smis_set) # copy
    in_basic = []
    for i, smi in enumerate(in_smis):
        in_basic.append(in_basic_temp[in_mols_temp.index(smi)])

    gen_basic_props = decoy_utils.calc_dataset_props_basic(gen_mols, verbose=True)

    print('Scale properties based on in_mols props')
    active_props_scaled_all = []
    decoy_props_scaled_all = []

    active_min_all = []
    active_max_all = []
    active_scale_all = []

    active_props = in_props_temp
    print('Exclude errors from min/max calc')
    act_prop = np.array(active_props)

    active_maxes = np.amax(act_prop, axis=0)
    active_mins = np.amin(act_prop, axis=0)

    active_max_all.append(active_maxes)
    active_min_all.append(active_mins)

    scale = []
    for (a_max, a_min) in zip(active_maxes,active_mins):
        if a_max != a_min:
            scale.append(a_max - a_min)
        else:
            scale.append(a_min)
    scale = np.array(scale)
    scale[scale == 0.0] = 1.0
    active_scale_all.append(scale)
    active_props_scaled = (active_props - active_mins) / scale
    active_props_scaled_all.append(active_props_scaled)

    # Calc SA scores
    in_sa_temp = [sascorer.calculateScore(mol) for mol in set_in_mols]
    in_smis_temp = list(set(in_smis))
    in_sa = []
    for i, smi in enumerate(in_smis):
        in_sa.append(in_sa_temp[in_smis_temp.index(smi)])
    gen_sa_props = [sascorer.calculateScore(mol) for mol in gen_mols]

    print('Calc Morgan fingerprints')
    in_fps = []
    for i, mol in enumerate(in_mols):
        in_fps.append(
            AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024))
    gen_fps = []
    for i, mol in enumerate(gen_mols):
        gen_fps.append(
            AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024))

    print('Calc DG scores')
    dg_scores, dg_ids = decoy_utils.dg_score_rev(set_in_mols, gen_mols)

    print('Calc LADS scores')
    lads_scores = decoy_utils.lads_score_v2(set_in_mols, gen_mols)
    
    print('Construct dictionary of results')
    results_dict = {}
    for i in range(len(in_smis)):
        # Get scaling
        in_props_scaled = (in_props[i] - active_min_all) / active_scale_all
        gen_props_scaled = (np.array(gen_props[i]) - active_min_all) / active_scale_all
        prop_diff = np.linalg.norm(np.array(in_props_scaled)-np.array(gen_props_scaled))

        # Get basic props diff
        basic_diff = np.sum(abs(np.array(in_basic[i])-np.array(gen_basic_props[i])))

        if in_smis[i] in results_dict:
            sim = DataStructs.TanimotoSimilarity(in_fps[i], gen_fps[i])
            results_dict[in_smis[i]].append(
                [in_smis[i], gen_smis[i], in_props[i], gen_props[i], prop_diff, 
                 sim, basic_diff, abs(gen_sa_props[i]-in_sa[i]), 
                 dg_scores[i], lads_scores[i], gen_mols[i]])
        else:
            sim = DataStructs.TanimotoSimilarity(in_fps[i], gen_fps[i])
            results_dict[in_smis[i]] = [
                [in_smis[i], gen_smis[i], in_props[i], gen_props[i], prop_diff, 
                 sim, basic_diff, abs(gen_sa_props[i]-in_sa[i]), 
                 dg_scores[i], lads_scores[i], gen_mols[i]] ]

    print('Get decoy matches')
    results = []
    results_success_only = []
    sorted_mols_success = []
    for key in results_dict:
        # Set initial criteria - Note most of these are relatively weak
        prop_max_diff = 5
        max_basic_diff = 3
        max_sa_diff = 1.51
        max_dg_score = 0.35
        max_lads_score = 0.2# 5# 0.1
        while True:
            count_success = sum([i[4]<prop_max_diff \
                                 and i[6]<max_basic_diff and i[7]<max_sa_diff \
                                 and i[8]<max_dg_score and i[9]<max_lads_score \
                                     for i in results_dict[key][0:max_idx_cmpd]])
            # Adjust criteria if not enough successes
            if count_success < num_cand_dec_per_act and max_dg_score<1:
                #print("Widening search", count_success)
                prop_max_diff *= 1.1
                max_basic_diff += 1
                max_sa_diff *= 1.1
                max_dg_score *= 1.1
                max_lads_score *= 1.1
            else:
                #print("Reached threshold", count_success)
                # Sort by sum of LADS and property difference (smaller better)
                sorted_mols_success.append(
                    [(i[0], i[1], i[4], i[9], i[4]+i[9], i[10]) \
                     for i in sorted(results_dict[key][0:max_idx_cmpd], 
                                     key=lambda i: i[4]+i[9], reverse=False)   
                    if i[4]<prop_max_diff \
                        and i[6]<max_basic_diff and i[7]<max_sa_diff \
                            and i[8]<max_dg_score and i[9]<max_lads_score])
                #assert count_success == len(sorted_mols_success[-1])
                break

    print('Choose decoys')
# =============================================================================
#     active_smis_gen = []
# =============================================================================
    decoy_smis_gen = set()

    embed_fails = 0
    dupes_wanted = 0
    for act_res in sorted_mols_success:
        count = 0
        # Greedy selection based on sum of LADS score and property difference (smaller better)
        for ent in act_res:
            # Check can gen conformer
            if ent[1] not in decoy_smis_gen: # Check conf and not a decoy for another ligand
                decoy_smis_gen.update([ent[1]])
                count +=1
                if count >= num_dec_per_act:
                    break
            elif ent[1] in decoy_smis_gen:
                dupes_wanted +=1
            else:
                embed_fails += 1
# =============================================================================
#         active_smis_gen.append(act_res[0][0])
# =============================================================================
    decoy_smis_gen = list(decoy_smis_gen)
    decoy_mols_gen = [Chem.MolFromSmiles(smi) for smi in decoy_smis_gen]
# =============================================================================
#     active_mols_gen = [Chem.MolFromSmiles(smi) for smi in active_smis_gen]
# =============================================================================
    active_mols_gen = set_in_mols
    dataset = 'dude'
    print('Calc props for chosen decoys')
    if dataset == "dude_ext":
        actives_feat = decoy_utils.calc_dataset_props_dude_extended(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_dude_extended(decoy_mols_gen, verbose=True)
    elif dataset == "dekois":
        actives_feat = decoy_utils.calc_dataset_props_dekois(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_dekois(decoy_mols_gen, verbose=True)
    elif dataset == "MUV":
        actives_feat = decoy_utils.calc_dataset_props_muv(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_muv(decoy_mols_gen, verbose=True)
    elif dataset == "ALL":
        actives_feat = decoy_utils.calc_dataset_props_all(active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_all(decoy_mols_gen, verbose=True)
    elif dataset == "dude":
        actives_feat = decoy_utils.calc_dataset_props_dude(active_mols_gen)
        decoys_feat = decoy_utils.calc_dataset_props_dude(decoy_mols_gen)
    else:
        print("Incorrect dataset")
        exit()

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
    
    # Save intermediate performance results in unique file
    #with open(output_loc+'results_'+f+'.csv', 'w') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #    writer.writerow(dec_results)

    print('Save decoy mols')
    output_name = output_loc + \
        f'/{target}_{idx}_selected_{num_dec_per_act}_{num_cand_dec_per_act}.smi'
    with open(output_name, 'w') as outfile:
        for i, smi in enumerate(decoy_smis_gen):
            outfile.write(set_in_smis[i//num_dec_per_act] + ' ' + smi + '\n')
    print(dec_results)
    GM = np.mean(dec_results[7+1:7+1+3])
    print(f'GM: {GM:.4f}')
    dec_results.append(GM)
    return dec_results

import argparse
if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='./eval/results/')
    parser.add_argument('--dataset_name', default='dude')
    parser.add_argument('--max_idx_cmpd', default=10000)
    parser.add_argument('--target')
    N = 50#50
    # parser.add_argument('--times')
    args = parser.parse_args()
    
    target = args.target
    # times = int(args.times)
    output_loc = args.output_path
    dataset = args.dataset_name
    # num_dec_per_act = args.num_decoys_per_active * times
    num_cand_dec_per_act = N# 1
    num_dec_per_act = N# 30
    max_idx_cmpd = int(args.max_idx_cmpd)
    min_active_size = 10# int(args.min_active_size)

    output_name = f'./eval/results/{target}_-1_selected_{num_dec_per_act}_{num_cand_dec_per_act}.smi'
    df = []
    for idx in range(100):
        try:
            df.append(pd.read_csv(f'./eval/results/{target}_{idx}_selected_30_500.smi', 
                                  header=None, sep=' ', names=['ace', 'dec']))
        except:
            break
    print('max idx', idx)
    df = pd.concat(df)
    df = df.drop_duplicates()
    df.to_csv(output_name, index=False, header=None, sep=' ')
    # Select decoys and evaluate
    from select_new import run
    run(output_name, output_name, topk=N*5)
    
    with open(output_loc+f'/{target}_results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        print('*'*100)
        results = select_and_evaluate_decoys(
                f=output_name.split('/')[-1], target=target, idx=idx, 
                file_loc=output_loc, 
                output_loc=output_loc, dataset=dataset, 
                num_cand_dec_per_act=num_cand_dec_per_act, num_dec_per_act=num_dec_per_act, 
                max_idx_cmpd=max_idx_cmpd)
        writer.writerow(results) 
        
