import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
import pickle
from rdkit import Chem
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--target', default='INHA')
parser.add_argument('--idx')
config = parser.parse_args()
idx = config.idx
target = config.target# .split("_")[0]


result = np.array(pickle.load(open(f'./eval/results/{target}_{idx}_result.pkl', 'rb')))
print(result.shape)
act = list(set(pickle.load(open(f'./data/content_test_{target.split("_")[0]}.pkl', 'rb'))))

df = []
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        smi = result[i][j]
        if Chem.MolFromSmiles(smi) is not None:
            df.append([act[i], smi])
df = pd.DataFrame(df, columns=['act', 'dec'])
print(df.shape)

df.dec = df.dec.apply(lambda x: x.replace('\n', ''))
df.to_csv(f'./eval/results/{target}_{idx}_tmp.csv', index=False, header=None, sep=' ')
