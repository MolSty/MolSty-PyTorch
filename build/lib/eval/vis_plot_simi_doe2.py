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
min_active_size = 10
def select_and_evaluate_decoys(data):
    dec_results = []
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
    
    # Save intermediate performance results in unique file
    #with open(output_loc+'results_'+f+'.csv', 'w') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #    writer.writerow(dec_results)
    return dec_results

df = pd.read_csv('./INHA_3_tmp.csv', header=None, names=['act', 'dec'],
                 sep=' ')
def get_mol_features(mol):
    try:
        fp3 = AllChem.GetMACCSKeysFingerprint(mol).ToBitString()
        fp3 = [int(item) for item in fp3]
        if np.sum(fp3) == 0:
            return fp3
        fp3 = list((np.array(fp3)/(np.sum(fp3)**0.5)).astype(np.float16))
        return np.array(fp3)
    except:
        return -1

# df['dec_fp'] = df['dec'].apply(lambda x: get_mol_features(Chem.MolFromSmiles(x)))
df = df.sample(frac=0.1).reset_index(drop=True)
df2 = df.copy()
# =============================================================================
# 
# =============================================================================
import torch
from moses.models_storage import ModelsStorage
from moses.latentgan.model import load_model
MODELS = ModelsStorage()

model_config = torch.load('../temp')
model_vocab = torch.load('../vocab')
model_state = torch.load('../t_020.pt')
model = MODELS.get_model_class('latentgan')(model_vocab, model_config)
model.load_state_dict(model_state)
model = model.cuda()
model.eval()

model.model_loaded = True
_, smi2vec = load_model()

act2fp = set(df['act'])
act2fp = {smi:model.heteroencoder.encode(smi2vec([smi])) for smi in act2fp}
act2fp = {key: np.array(item)/(np.sum(item**2)**0.5) for key, item in act2fp.items()}

df2['simi'] = df2['dec'].apply(lambda x: model.heteroencoder.encode(smi2vec([x])))
df2['simi'] = df2['simi'].apply(lambda x:np.array(x)/(np.sum(x**2)**0.5))

df2['simi'] = df2[['act', 'simi']].values.tolist()
df2['simi'] = df2['simi'].apply(lambda x: np.sum(act2fp[x[0]]*x[1]))
print(df2.head())
# =============================================================================
# 
# =============================================================================
# =============================================================================
# df2['simi'] = df2[['act', 'dec_fp']].values.tolist()
# df2['simi'] = df2['simi'].apply(lambda x: np.sum(act2fp[x[0]]*x[1]) if type(x[1])!=int else 0)
# =============================================================================
# df2 = df2.loc[df2['dec_fp'].apply(lambda x: 1 if type(x)!=int else 0)==1].reset_index(drop=True)
MIN = df2['simi'].min()
MAX = df2['simi'].max()

df2['simi'] = df2['simi'].apply(lambda x: (x-MIN)//((MAX-MIN)*0.1))
df2['simi'] = df2['simi'].astype(int)
results_lst = []
for i in range(10):
    tmp = df2.loc[df2.simi==i].reset_index(drop=True)
    if tmp.shape[0] < 100:
        continue
    results_lst.append([i]+select_and_evaluate_decoys(tmp[['act', 'dec']].values.tolist()))

import matplotlib.pyplot as plt
print(results_lst)
results_lst = np.array(results_lst)
x = results_lst[:, 0] * 0.1
plt.plot(x, results_lst[:, 7],'o-',label="DOE")
plt.plot(x, results_lst[:, 8],'o-',label="LADS")
plt.plot(x, results_lst[:, 6],'o-',label="AUC")
plt.plot(x, results_lst[:, 9],'o-',label="DG")
# plt.xticks(rotation=45)
plt.legend()
plt.savefig('./fig.jpg')
plt.show()