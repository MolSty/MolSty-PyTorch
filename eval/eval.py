import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from rdkit import Chem
import sascorer
from rdkit.Chem import AllChem
from pss import get_pss_from_smiles
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='TOX', choices=['TOX', 'SA'])
args = parser.parse_args()
target = args.target

def get_sa(x):
    try:
        return sascorer.calculateScore(Chem.MolFromSmiles(x))
    except:
        return -1

def get_mol_features(mol):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    try:
        fp3 = AllChem.GetMACCSKeysFingerprint(mol).ToBitString()
        fp3 = np.array([int(item) for item in fp3])
        if np.sum(fp3) == 0:
            return fp3
        fp3 = (fp3/(np.sum(fp3**2)**0.5))
        return list(fp3)
    except:
        return 0

fn = f'./eval/results/{target}_results.smi'
print(fn)
if target == 'SA':
    df = pd.read_csv(fn, sep=' ', 
                     header=None, names=['content', 'gene'])
    df['content_SA'] = df['content'].apply(lambda x: get_sa(x))
    df['gene_SA'] = df['gene'].apply(lambda x: get_sa(x))
    df = df.loc[df['gene_SA']!=-1].reset_index(drop=True)
else:
    df = pd.read_csv(fn, sep=' ', 
                     header=None, names=['content', 'gene'])
    pred = pd.read_csv('./eval/results/predict_TOX.csv').rename(
        columns={'smiles':'gene', 'pred_0':'gene_TOX'})[['gene', 'gene_TOX']]
    df = df.merge(pred, on='gene', how='inner')
    pred = pd.read_csv('./eval/zinc_all_alerts_pred.csv', 
                       usecols=['smiles', 'pred']).rename(
        columns={'smiles':'content', 
                 'pred':'content_TOX'})[['content', 'content_TOX']]
    df = df.merge(pred, on='content', how='left')
df = df.drop_duplicates(['content', 'gene'])
df['tmp'] = df[['content', 'gene']].values.tolist()
print(df.shape)
pss = get_pss_from_smiles(df['content'].values, df['gene'].values)
df['PSS'] = pss.mean(0)
df['content_fp'] = df['content'].apply(
    lambda x: get_mol_features(x))
df['gene_fp'] = df['gene'].apply(
    lambda x: get_mol_features(x))
df['similarity'] = df[['content_fp', 'gene_fp']].values.tolist()
df['similarity'] = df['similarity'].apply(
    lambda x: np.sum(np.array(x[0])*np.array(x[1])))

# higher is better
def GM_score(lst):
    x = 1
    for item in lst:
        x *= item
    return (x)**(1/len(lst))

def SR_score(gene):
    T_PSS = 0.75#0.8
    if target == 'TOX':
        T_style = 0.1#0.1
    else:
        T_style = 2.5
    return gene.loc[np.logical_and(
        gene[f'gene_{target}']<T_style,
        gene['PSS']>T_PSS)].shape[0] / gene.shape[0]
PSS = np.mean(pss)
Improvement = np.mean(df[f"content_{target}"]-df[f"gene_{target}"])
Similarity = df["similarity"].mean()
SR = SR_score(df)
GM = GM_score([Improvement, PSS, SR])
print(f'{target} (raw): {df[f"content_{target}"].mean():.3f},',
      f'{target} (generate): {df[f"gene_{target}"].mean():.3f},',
      f'PSS: {PSS:.3f}, ',
      f'Improvement: {Improvement:.3f}',
      f'SR: {SR:.3f}',
      f'Similarity: {Similarity:.3f},',
      f'GM: {GM:.3f}')
