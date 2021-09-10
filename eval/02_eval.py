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
from pss import get_pss_from_smiles

from joblib import Parallel, delayed
from docopt import docopt
def get_sa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol)
    except:
        return -1

# Worker function
def select_and_evaluate_decoys(f, target, file_loc='./', output_loc='./'):
    print("Processing: ", f)
    dec_results = [f]
    # Read data
    data = decoy_utils.read_paired_file(file_loc+f)
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    tmp = [Chem.MolFromSmiles(d[0]) for d in data]
    data = [d for idx, d in enumerate(data) if tmp[idx] is not None \
            and tmp[idx].GetNumHeavyAtoms()>10]
    data = pd.DataFrame(data, columns=['act', 'dec'])
    if target == 'SA':
        data['style'] = data['dec'].apply(lambda x: get_sa(x))
        data['style'] = (5 - data['style']) / 3
    else:
        style = pd.read_csv('./eval/results/predict_TOX.csv')
        style = style.rename(
            columns={'smiles':'dec', 
                     'pred_0':'style'})[['dec', 'style']]
        data = data.merge(style, on='dec', how='inner')
        data['style'] = 1 - data['style']
    pss = get_pss_from_smiles(
        data['act'].values, data['dec'].values)
    data['pss'] = pss.mean(0)
    data['score'] = data['pss'] + data['style']
    result = []
    for key, tmp_df in data.groupby('act'):
        tmp_df = tmp_df.sort_values('score', ascending=False)
        tmp_df = tmp_df.reset_index(drop=True)
        for i in range(min([1, tmp_df.shape[0]])):
            result.append([key, tmp_df['dec'].values[i]])
    result = pd.DataFrame(result, columns=['act', 'dec'])
    output_name = output_loc + \
        f'/{target}_results.smi'
    
    result.to_csv(output_name, index=False, header=None, sep=' ')
    

import argparse
if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='tmp.csv')
    parser.add_argument('--output_path', default='./eval/results/')
    parser.add_argument('--target')
    args = parser.parse_args()
    
    target = args.target
    file_loc = args.data_path
    file_loc = f'{target}_{file_loc}'
    output_loc = args.output_path

    select_and_evaluate_decoys(
        f=file_loc, 
        target=target, 
        file_loc=output_loc, 
        output_loc=output_loc)
