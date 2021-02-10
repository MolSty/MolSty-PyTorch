import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs

def calc_charges(mol):
    positive_charge, negative_charge = 0, 0
    for atom in mol.GetAtoms():
        charge = float(atom.GetFormalCharge())
        positive_charge += max(charge, 0)
        negative_charge -= min(charge, 0)

    return positive_charge, negative_charge

def calc_props_all(mol):
    try:
        if type(mol) is str:
            mol = Chem.MolFromSmiles(mol)
        # Calculate properties and store in dict
        prop_dict = {}
        # molweight
        prop_dict.update({'mol_wg': Descriptors.MolWt(mol)})
        # logP
        prop_dict.update({'log_p': Chem.Crippen.MolLogP(mol)})
        # HBA
        prop_dict.update({'hba': Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)})
        # HBD
        prop_dict.update({'hbd': Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)})
        # rotatable bonds
        prop_dict.update({'rot_bnds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)})
        # Formal (net) charge
        prop_dict.update({'net_charge': Chem.rdmolops.GetFormalCharge(mol)})
        
        prop_dict.update({'ring count': Chem.rdMolDescriptors.CalcNumRings(mol)})
        prop_dict.update({'TPSA': Chem.rdMolDescriptors.CalcTPSA(mol)})
        
        prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], 
                      prop_dict['hba'], prop_dict['hbd'], 
                      prop_dict['rot_bnds'], prop_dict['net_charge'],
                      prop_dict['ring count'], prop_dict['TPSA']]

        return prop_array

    except:
        return [0, 0, 0, 0, 0, 0, 0, 0]
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
        pss.append(tmp)
    return np.array(pss)

def get_pss_from_smiles(acts, decs):
    act_tmp = [calc_props_all(smi) for smi in acts]
    dec_tmp = [calc_props_all(smi) for smi in decs]
    
    act_prop, dec_prop = [], []
    for i in range(len(act_tmp)):
        # if np.sum(act_tmp[i]) != 0 and np.sum(dec_tmp[i]) != 0:
        act_prop.append(act_tmp[i])
        dec_prop.append(dec_tmp[i])
    pss = cal_pss(act_prop, dec_prop)
    return pss

if __name__ == '__main__':
    df = pd.read_csv('../tmp.smi', header=None,  
                     # ../SA_RAND.smi
                     names=['act', 'dec'], sep=' ')
    pss = get_pss_from_smiles(df.act.values, df.dec.values)
    print(f'pss: {np.mean(pss):.3f}')
