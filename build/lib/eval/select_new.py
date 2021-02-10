import pandas as pd
pd.set_option('display.max_columns', 12)
from sklearn.model_selection import train_test_split
from decoy_utils import parallel_func
from rdkit import Chem
import pickle
import numpy as np
from rdkit.Chem import Descriptors
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*') 
def calc_props_dude(mol): # smiles
    try:
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

        prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], prop_dict['hba'],
                      prop_dict['hbd'], prop_dict['rot_bnds'], prop_dict['net_charge']]
        
        prop_array = (np.array(prop_array) - np.array([200,-4,2,2,0,-3])) / np.array([500,10,12,12,20,6])
        return prop_array / ((prop_array**2).sum())**0.5

    except:
        return None
        # return [0, 0, 0, 0, 0, 0]

    


def run(input_path, output_path, topk=1):
    df = pd.read_csv(input_path, header=None, names=['act', 'dec'], sep=' ')
    act_smi = list(set(df.act))
    dec_smi = list(set(df.dec))
    print(input_path)
    print(output_path)
    print(len(act_smi), len(dec_smi))
    
    act_mol = [Chem.MolFromSmiles(smi) for smi in act_smi]
    dec_mol = [Chem.MolFromSmiles(smi) for smi in dec_smi]
    
    act_prop = parallel_func(act_mol, calc_props_dude)
    dec_prop = parallel_func(dec_mol, calc_props_dude)
    dec_prop = np.array([item for item in dec_prop if item is not None])
    dec_prop = dec_prop.T
    
    print(f'select top-{topk}')
    act = []
    dec = []
    for i, a in enumerate(act_prop):
        smi = a.dot(dec_prop)
        idxs = np.argsort(smi)[::-1]
        for k in range(topk):
            j = idxs[k]
            act.append(act_smi[i])
            dec.append(dec_smi[j])
        
    df = pd.DataFrame(act, columns=['act'])
    df['dec'] = dec
    df.to_csv(output_path, index=False, header=None, sep=' ')
    print(df.shape)


if __name__ == '__main__':
    import pandas as pd
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import MolFromSmiles
    df = pd.read_csv('../data/dataset_v1.csv')
    smis = df.SMILES.tolist()[:10000]
    mols = parallel_func(smis, MolFromSmiles)
    # mols = [Chem.MolFromSmiles(smi) for smi in smis]
    arr = parallel_func(mols, calc_props_dude)
    print(arr.shape)
    
# =============================================================================
#     for mol in tqdm(mols):
#         calc_props_dude(mol)
# =============================================================================
