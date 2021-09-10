import warnings
warnings.filterwarnings('ignore')

from rdkit.Chem import Descriptors
# =============================================================================
# def calc_props_dude(mol): # smiles
#     try:
#         # Calculate properties and store in dict
#         prop_dict = {}
#         # molweight
#         prop_dict.update({'mol_wg': Descriptors.MolWt(mol)})
#         # logP
#         prop_dict.update({'log_p': Chem.Crippen.MolLogP(mol)})
#         # HBA
#         prop_dict.update({'hba': Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)})
#         # HBD
#         prop_dict.update({'hbd': Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)})
#         # rotatable bonds
#         prop_dict.update({'rot_bnds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)})
#         # Formal (net) charge
#         prop_dict.update({'net_charge': Chem.rdmolops.GetFormalCharge(mol)})
# 
#         prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], prop_dict['hba'],
#                       prop_dict['hbd'], prop_dict['rot_bnds'], prop_dict['net_charge']]
# 
#         return prop_array
# 
#     except:
#         return [0, 0, 0, 0, 0, 0]
# =============================================================================

def calc_props_all(mol):
    try:
        props = []
             
        ### MUV properties ###
        # num atoms (incl. H)
        props.append(mol.GetNumAtoms(onlyExplicit=False))
        # num heavy atoms
        props.append(mol.GetNumHeavyAtoms())
        # num boron
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#5]"), maxMatches=mol.GetNumAtoms())))
        # num carbons
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"), maxMatches=mol.GetNumAtoms())))
        # num nitrogen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7]"), maxMatches=mol.GetNumAtoms())))
        # num oxygen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]"), maxMatches=mol.GetNumAtoms())))
        # num fluorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#9]"), maxMatches=mol.GetNumAtoms())))
        # num phosphorus
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#15]"), maxMatches=mol.GetNumAtoms())))
        # num sulfur
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]"), maxMatches=mol.GetNumAtoms())))
        # num chlorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#17]"), maxMatches=mol.GetNumAtoms())))
        # num bromine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#35]"), maxMatches=mol.GetNumAtoms())))
        # num iodine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#53]"), maxMatches=mol.GetNumAtoms())))

        # logP
        props.append(Chem.Crippen.MolLogP(mol))
        # HBA
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol))
        # HBD
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol))
        # ring count
        props.append(Chem.rdMolDescriptors.CalcNumRings(mol))
        # Stereo centers
        props.append(len(Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True)))
        
        ### DEKOIS properties (additional) ###
        # molweight
        props.append(Descriptors.MolWt(mol))
        # aromatic ring count
        props.append(Chem.rdMolDescriptors.CalcNumAromaticRings(mol))
        # rotatable bonds
        props.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol))
        # Pos, neg charges
        pos, neg = calc_charges(mol)
        props.append(pos)
        props.append(neg)
        
        ### DUD-E extended (additional) ###
        # Formal (net) charge
        props.append(Chem.rdmolops.GetFormalCharge(mol))
        # Topological polar surface area
        props.append(Chem.rdMolDescriptors.CalcTPSA(mol))
        
        ### Additional
        # QED
        props.append(Chem.QED.qed(mol))
        # SA score
        
        return props

    except:
        return [0]*25

def calc_charges(mol):
    positive_charge, negative_charge = 0, 0
    for atom in mol.GetAtoms():
        charge = float(atom.GetFormalCharge())
        positive_charge += max(charge, 0)
        negative_charge -= min(charge, 0)

    return positive_charge, negative_charge
def calc_props_dekois(mol):
    # Create RDKit mol
    try:
        mol = Chem.AddHs(mol)
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
        # aromatic ring count
        prop_dict.update({'ring_ct': Chem.rdMolDescriptors.CalcNumAromaticRings(mol)})
        # rotatable bonds
        prop_dict.update({'rot_bnds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)})
        # Formal charges
        pos, neg = calc_charges(mol)
        prop_dict.update({'pos_charge': pos})
        prop_dict.update({'neg_charge': neg})

        prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], prop_dict['hba'],
                      prop_dict['hbd'], prop_dict['ring_ct'], prop_dict['rot_bnds'],
                      prop_dict['pos_charge'], prop_dict['neg_charge']]

        return prop_array

    except:
        return [0, 0, 0, 0, 0, 0, 0, 0]


from sklearn.metrics import roc_curve
def doe_score(actives, decoys):
    # Lower is better 
    all_feat = list(actives) + list(decoys)
    up_p = np.percentile(all_feat, 95, axis=0)
    low_p = np.percentile(all_feat, 5, axis=0)
    norms = up_p - low_p
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1.

    active_norm = [act/norms for act in actives]
    decoy_norm = [dec/norms for dec in decoys]
    all_norm = active_norm + decoy_norm

    active_embed = []
    labels = [1] * (len(active_norm)-1) + [0] * len(decoy_norm)
    for i, act in enumerate(active_norm):
        comp = list(all_norm)
        del comp[i]
        dists = [100 - np.linalg.norm(c-act) for c in comp] # arbitrary large number to get scores in reverse order
        fpr, tpr, _ = roc_curve(labels, dists)
        fpr = fpr[::]
        tpr = tpr[::]
        a_score = 0
        for i in range(len(fpr)-1):
            a_score += (abs(0.5*( (tpr[i+1]+tpr[i])*(fpr[i+1]-fpr[i]) - (fpr[i+1]+fpr[i])*(fpr[i+1]-fpr[i]) )))
        active_embed.append(a_score)

    #print(np.average(active_embed))
    return np.array(active_embed)


from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import DataStructs
def dg_score(active_mols, decoy_mols):
    # Similar to DEKOIS
    # Lower is better (less like actives), higher is worse (more like actives)
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,3,useFeatures=True) \
                  for mol in active_mols] # Roughly FCFP_6
    decoys_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,3,useFeatures=True) \
                  if mol is not None else None for mol in decoy_mols] # Roughly FCFP_6

    closest_sims = []
    closest_sims_id = []
    for active_fp in active_fps:
        active_sims = []
        for decoy_fp in decoys_fps:
            active_sims.append(DataStructs.TanimotoSimilarity(active_fp, decoy_fp) \
                               if decoy_fp is not None else 0)
        closest_sims.append(max(active_sims))
        closest_sims_id.append(np.argmax(active_sims))

    return np.array(closest_sims), np.array(closest_sims_id)
import pickle
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE

target = 'FA7' # THB
acts_mols = list(filter(lambda x: x is not None, 
                        Chem.SDMolSupplier(f'../data/{target}_actives_final.sdf')))
acts_mols = list(set([Chem.MolToSmiles(mol) for mol in acts_mols]))
acts_mols = [Chem.MolFromSmiles(smi) for smi in acts_mols]
# decs_smiles = pd.read_csv('./test.csv')['dec'].values[:256]
decs_mols = list(filter(lambda x: x is not None, 
                        Chem.SDMolSupplier(f'../data/{target}_decoys_final.sdf')))
decs_mols = list(set([Chem.MolToSmiles(mol) for mol in decs_mols]))
decs_mols = [Chem.MolFromSmiles(smi) for smi in decs_mols]

active_props = [calc_props_all(mol) for mol in acts_mols]
decoy_props = [calc_props_all(mol) for mol in decs_mols]

# =============================================================================
# X = np.concatenate([active_props, decoy_props], 0)
# y = np.array([0]*len(active_props)+[1]*len(decoy_props))
# tsne = TSNE(n_components=2)
# for i in range(X.shape[1]):
#     X[:, i] = (X[:, i] - X[:, i].min()) / X[:, i].ptp()
# result = tsne.fit_transform(X)
# 
# idx = np.where(y==1)[0]
# plt.plot(result[idx][:, 0], result[idx][:, 1], '.', label='DUD-E')
# 
# idx = np.where(y==0)[0]
# plt.plot(result[idx][:, 0], result[idx][:, 1], '.', label='Active')
# plt.legend()
# plt.show()
# =============================================================================
# =============================================================================
# 
# =============================================================================
gens_smiles = pd.read_csv(f'../result/{target}/selected_30_500.smi', 
                          sep=' ', header=None, names=['act', 'dec'])
gens_smiles = set(gens_smiles.dec)
# acts_mols = set(gens_smiles.act)
gens_mols = []
for smiles in gens_smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            gens_mols.append(mol)
    except:
        pass
gene_props = [calc_props_all(mol) for mol in gens_mols]

# =============================================================================
# X2 = np.concatenate([active_props, gene_props], 0)
# y2 = np.array([0]*len(active_props)+[1]*len(gene_props))
# tsne2 = TSNE(n_components=2)
# for i in range(X2.shape[1]):
#     X2[:, i] = (X2[:, i] - X2[:, i].min()) / X2[:, i].ptp()
# result2 = tsne2.fit_transform(X2)
# 
# idx = np.where(y2==1)[0]
# plt.plot(result2[idx][:, 0], result2[idx][:, 1], '.', label='Ours')
# 
# idx = np.where(y2==0)[0]
# plt.plot(result2[idx][:, 0], result2[idx][:, 1], '.', label='Active')
# plt.legend()
# plt.show()
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# X3 = np.concatenate([active_props, decoy_props, gene_props], 0)
# y3 = np.array([0]*len(active_props)+[1]*len(decoy_props)+[2]*len(gene_props))
# tsne3 = TSNE(n_components=2)
# for i in range(X3.shape[1]):
#     X3[:, i] = (X3[:, i] - X3[:, i].min()) / X3[:, i].ptp()
# result3 = tsne3.fit_transform(X3)
# 
# idx = np.where(y3==1)[0]
# plt.plot(result3[idx][:, 0], result3[idx][:, 1], '.', label='DUD-E')
# 
# idx = np.where(y3==2)[0]
# plt.plot(result3[idx][:, 0], result3[idx][:, 1], '.', label='Ours')
# 
# idx = np.where(y3==0)[0]
# plt.plot(result3[idx][:, 0], result3[idx][:, 1], 'r.', label='Active')
# plt.legend()
# plt.show()
# =============================================================================
# =============================================================================
# 
# =============================================================================
gens_smiles = pd.read_csv(f'../result/{target}/selected_30_500.smi', 
                          sep=' ', header=None, names=['act', 'dec'])
temp_act = gens_smiles.act[61]
gens_smiles = gens_smiles.loc[gens_smiles.act==temp_act]
gens_smiles = list(set(gens_smiles.dec)) + [temp_act]
gens_mols = []
for smiles in gens_smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            gens_mols.append(mol)
    except:
        pass
gene_props_single = [calc_props_all(mol) for mol in gens_mols]
# =============================================================================
# X4 = np.concatenate([active_props, decoy_props, gene_props, gene_props_single], 0)
# y4 = np.array([0]*len(active_props)+[1]*len(decoy_props)+[2]*len(gene_props)+[3]*len(gene_props_single))
# tsne4 = TSNE(n_components=2)
# for i in range(X4.shape[1]):
#     X4[:, i] = (X4[:, i] - X4[:, i].min()) / X4[:, i].ptp()
# result4 = tsne4.fit_transform(X4)
# 
# idx = np.where(y4==1)[0]
# plt.plot(result4[idx][:, 0], result4[idx][:, 1], '.', label='DUD-E')
# 
# idx = np.where(y4==2)[0]
# plt.plot(result4[idx][:, 0], result4[idx][:, 1], '.', label='Ours')
# 
# idx = np.where(y4==0)[0]
# plt.plot(result4[idx][:, 0], result4[idx][:, 1], 'r.', label='Active')
# 
# idx = [np.where(y4==3)[0][-1]]
# plt.plot(result4[idx][:, 0], result4[idx][:, 1], 'k^', label='target')
# 
# idx = np.where(y4==3)[0][:-1]
# plt.plot(result4[idx][:, 0], result4[idx][:, 1], 'kx', label='single')
# plt.legend()
# plt.show()
# =============================================================================
# =============================================================================
# 
# =============================================================================
df = pd.read_csv(
    f'C:/Users/戴尔/Desktop/DeepCoy_decoys/DeepCoy-DUDE-SMILES/dude-target-{target.lower()}-decoys-final.txt',
                 sep=' ', header=None, names=['act', 'dec'])
decs_deepcoy_mols = list(set(df.dec.tolist()))
decs_deepcoy_mols = [Chem.MolFromSmiles(smi) for smi in decs_deepcoy_mols]
decoy_deepcoy_props = [calc_props_all(mol) for mol in decs_deepcoy_mols]
X5 = np.concatenate([active_props, decoy_props, gene_props, gene_props_single, decoy_deepcoy_props], 0)
y5 = np.array([0]*len(active_props)+[1]*len(decoy_props)+\
              [2]*len(gene_props)+[3]*len(gene_props_single)+[4]*len(decoy_deepcoy_props))
tsne5 = TSNE(n_components=2)
for i in range(X5.shape[1]):
    if np.isnan(X5[:, i][0]) or X5[:, i].ptp()==0:
        X5[:, i] = 0
    else:
        X5[:, i] = (X5[:, i] - X5[:, i].min()) / X5[:, i].ptp()
result5 = tsne5.fit_transform(X5)


idx = np.where(y5==2)[0]
plt.plot(result5[idx][:, 0], result5[idx][:, 1], '.', label='Ours')

idx = np.where(y5==1)[0]
plt.plot(result5[idx][:, 0], result5[idx][:, 1], '.', label='DUD-E')
# =============================================================================
# idx = [np.where(y5==3)[0][-1]]
# plt.plot(result5[idx][:, 0], result5[idx][:, 1], 'k^', label='target')
# 
# idx = np.where(y5==3)[0][:-1]
# plt.plot(result5[idx][:, 0], result5[idx][:, 1], 'kx', label='single')
# =============================================================================

idx = np.where(y5==4)[0]
plt.plot(result5[idx][:, 0], result5[idx][:, 1], '.', label='Deepcoy')

idx = np.where(y5==0)[0]
plt.plot(result5[idx][:, 0], result5[idx][:, 1], 'r.', label='Active')
plt.legend()
plt.show()