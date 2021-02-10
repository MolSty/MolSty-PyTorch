import pandas as pd
import numpy as np
from rdkit import Chem
from eval.decoy_utils import calc_props_all

def cal_pss(act_prop, dec_prop):
    act_prop = np.array(act_prop)
    dec_prop = np.array(dec_prop)
    assert len(act_prop) == len(dec_prop)
    pss = []
    for j in range(act_prop.shape[1]):
        MAX = act_prop[:, j].max()
        MIN = act_prop[:, j].min()
        tmp = []
        for i in range(act_prop.shape[0]):
            d1 = abs(act_prop[i][j] - MIN)
            d2 = abs(act_prop[i][j] - MAX)
            fj = max([d1, d2])
            theta_ij = 1 - abs(act_prop[i][j] - dec_prop[i][j])/fj if fj != 0 else 1
            tmp.append(theta_ij)
        pss.append(np.mean(tmp))
    return pss

def get_pss_from_smiles(acts, decs):
    act_tmp = [calc_props_all(smi) for smi in acts]
    dec_tmp = [calc_props_all(smi) for smi in decs]
    
    act_prop, dec_prop = [], []
    for i in range(len(act_tmp)):
        if np.sum(act_tmp[i]) != 0 and np.sum(dec_tmp[i]) != 0:
            act_prop.append(act_tmp[i])
            dec_prop.append(dec_tmp[i])
    pss = cal_pss(act_prop, dec_prop)
    return pss

if __name__ == '__main__':
    df = pd.read_csv('../SA_RAND_0_selected_30_500_output.smi', header=None,  
                     # ../SA_RAND.smi
                     names=['act', 'dec'], sep=' ')
    pss = get_pss_from_smiles(df.act.values, df.dec.values)
    print(f'pss: {np.mean(pss):.3f}')

'''
SA_RAND pss: 0.772
'''