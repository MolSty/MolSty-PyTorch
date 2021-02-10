from collections import Counter
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from rdkit import Chem
import torch.nn.functional as F

from moses.interfaces import MosesTrainer
from moses.utils import CharVocab, Logger
from .model import MolStyDataset
from .model import load_model
import pickle

class MolStyTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config
        self.latent_size = self.config.latent_vector_dim
        self._log2pi = None

    def build_initial_gaussian(self, instances):
        # instance: (K, B, T)
        assert len(instances.shape) == 3
        n_ins = instances.size(0)
        
        points = instances
        mu = points.mean(dim=0) # (B, latent_size)
        h = mu

        k_mu = mu.unsqueeze(0).repeat(n_ins, 1, 1) # (K, B, latent_size)
        std_sq = (points - k_mu).pow(2)
        std_sq = std_sq.sum(dim=0) / (n_ins-1) # unbiased estimator for variance

        return mu, std_sq, h    
    
    def extract_style_features(self, instances, IAF):
        mu, std_sq, h = self.build_initial_gaussian(instances)
        eps = torch.randn_like(mu)
        z0 = mu + eps * torch.sqrt(std_sq+1e-10) # (B, 2*H)
        log_qz0 = - 0.5*self._log2pi - 0.5 * torch.log(std_sq+1e-10) - 0.5 * eps**2

        # build flow
        z, log_det = IAF(z0, h)
        log_qz = log_qz0 + log_det
        return z, log_qz

    def _train_epoch(self, epoch, model, tqdm_data,
                     optimizer_disc=None, 
                     optimizer_gen=None):
        device = self.get_collate_device(model)
        if optimizer_disc is None:
            model.eval()
            optimizer_gen = None
        else:
            model.train()
        if self._log2pi is None:
            self._log2pi = torch.log(torch.tensor(
                2*np.pi, dtype=torch.float, device=device))

        postfix = {}
        disc_loss_batch = []
        g_loss_batch = []
        L_recon_batch = []
        L_cycle_batch = []

        for i, item in enumerate(tqdm_data):
            content_ins = torch.tensor([t[0] for t in item], dtype=torch.float64, device=device)
            style_ins = torch.tensor([t[1] for t in item], dtype=torch.float64, device=device)
            bs, hidden_size = content_ins.shape[0], content_ins.shape[1]
            n_ins = self.config.n_ins
            
            content_ins = content_ins.type(model.Tensor)
            style_ins = style_ins.type(model.Tensor)
            
            num = int(bs // n_ins * n_ins)
            content, _ = self.extract_style_features(
                content_ins[:num].view(n_ins, -1, hidden_size).contiguous(),
                IAF=self.IAF_s1)
            content = content.repeat(n_ins+1, 1)[:bs]
            style, _ = self.extract_style_features(
                style_ins[:num].view(n_ins, -1, hidden_size).contiguous(),
                IAF=self.IAF_s2)
            style = style.repeat(n_ins+1, 1)[:bs]
            
# =============================================================================
#             
# =============================================================================
            if optimizer_disc is not None:
                optimizer_disc.zero_grad()
            
            cc = content_ins
            sc = self.MLP_s1(content)
            cs = style_ins
            ss = self.MLP_s2(style)
            
            # 生成新的分子style&content排列组合形式
            ccsc = self.Generator(torch.cat([cc, sc], -1))
            ccss = self.Generator(torch.cat([cc, ss], -1))
            cssc = self.Generator(torch.cat([cs, sc], -1))
            csss = self.Generator(torch.cat([cs, ss], -1))
            
            style_1_validity = F.log_softmax(
                self.discriminator(torch.cat([content_ins, ccsc], 0)), dim=1)[:, 0]
            style_2_validity = F.log_softmax(
                self.discriminator(torch.cat([style_ins, csss], 0)), dim=1)[:, 1]
            fake_validity = F.log_softmax(
                self.discriminator(torch.cat([ccss, cssc], 0)), dim=1)[:, 2]
            
            # Gradient penalty
            gradient_penalty = model.compute_gradient_penalty(
                torch.cat([content_ins, ccsc], 0).data, 
                torch.cat([ccss, cssc], 0).data, self.discriminator)
            
            d_loss = - torch.mean(style_1_validity) - torch.mean(style_2_validity)\
                - torch.mean(fake_validity) + self.config.gp * gradient_penalty
                
            disc_loss_batch.append(d_loss.item())

            if optimizer_disc is not None:
                
                d_loss.backward()
                optimizer_disc.step()
                
                # Train the generator every n_critic steps
                if i % self.config.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_gen.zero_grad()
                    num = int(bs // n_ins * n_ins)
                    content, _ = self.extract_style_features(
                        content_ins[:num].view(n_ins, -1, hidden_size).contiguous(),
                        IAF=self.IAF_s1)
                    content = content.repeat(n_ins+1, 1)[:bs]
                    style, _ = self.extract_style_features(
                        style_ins[:num].view(n_ins, -1, hidden_size).contiguous(),
                        IAF=self.IAF_s2)
                    style = style.repeat(n_ins+1, 1)[:bs]
                    
                    cc = content_ins
                    sc = self.MLP_s1(content)
                    cs = style_ins
                    ss = self.MLP_s2(style)
                    
                    ccsc = self.Generator(torch.cat([cc, sc], -1))
                    ccss = self.Generator(torch.cat([cc, ss], -1))
                    cssc = self.Generator(torch.cat([cs, sc], -1))
                    csss = self.Generator(torch.cat([cs, ss], -1))
                    
                    style_1_validity = F.log_softmax(
                        self.discriminator(torch.cat([content_ins, ccsc], 0)), dim=1)[:, 0]
                    style_2_validity = F.log_softmax(
                        self.discriminator(torch.cat([style_ins, csss], 0)), dim=1)[:, 1]
                    fake_validity = F.log_softmax(
                        self.discriminator(torch.cat([ccss, cssc], 0)), dim=1)[:, 2]
            
                    # Gradient penalty
                    gradient_penalty = model.compute_gradient_penalty(
                        torch.cat([content_ins, ccsc], 0).data, 
                        torch.cat([ccss, cssc], 0).data, self.discriminator)
                    
                    g_loss = - torch.mean(style_1_validity) - torch.mean(style_2_validity)\
                        - torch.mean(fake_validity) + self.config.gp * gradient_penalty
                    
                    ccsc_decode_s = self.MLP_s1(ccsc) # mc
                    csss_decode_s = self.MLP_s2(csss) # ms
# =============================================================================
#                     
# =============================================================================
                    L_recon = 0.5 *  F.mse_loss(
                        self.Generator(torch.cat([cc, sc], -1)),
                        content_ins.detach())
                    L_recon += 0.5 * F.mse_loss(
                        self.Generator(torch.cat([cs, ss], -1)),
                        style_ins.detach())
                    g_loss += L_recon  #* 5
                    
                    L_recon2 = 0.5 * F.mse_loss(
                        self.Generator(torch.cat([cc, ss], -1)),
                        content_ins.detach())
                    L_recon2 +=0.5 * F.mse_loss(
                        self.Generator(torch.cat([cs, sc], -1)),
                        style_ins.detach())
                    g_loss += L_recon2  #* 5
                    
                    L_cycle =  0.5 * F.mse_loss(
                        ccsc_decode_s,
                        sc.detach())
                    L_cycle += 0.5 * F.mse_loss(
                        csss_decode_s,
                        ss.detach())
                    g_loss += L_cycle #* 10
                    

                    g_loss.backward()
                    optimizer_gen.step()
                    g_loss_batch.append(g_loss.item())
                    postfix['g_loss'] = np.mean(g_loss_batch)
                    
                    L_recon_batch.append(L_recon.item())
                    postfix['L_recon'] = np.mean(L_recon_batch)
                    L_cycle_batch.append(L_cycle.item())
                    postfix['L_cycle'] = np.mean(L_cycle_batch)
                    
            else:
                break
            postfix['d_loss'] = np.mean(disc_loss_batch)
            if optimizer_disc is not None:
                tqdm_data.set_postfix(postfix)
        postfix['mode'] = 'Eval' if optimizer_disc is None else 'Train'
        return postfix

    def _train(self, model, train_loader, val_loader=None, logger=None):

        device = model.device
        optimizer_disc = optim.Adam(
                list(self.discriminator.parameters()),
                                    lr=self.config.lr,
                                    betas=(self.config.b1, self.config.b2))
        optimizer_gen = optim.Adam(
                list(self.MLP_s1.parameters())+\
                list(self.MLP_s2.parameters())+\
                list(self.Generator.parameters())+\
                list(self.IAF_s1.parameters())+\
                list(self.IAF_s2.parameters()),
                                   lr=self.config.lr,
                                   betas=(self.config.b1, self.config.b2))
        scheduler_disc = optim.lr_scheduler.StepLR(optimizer_disc,
                                                   self.config.step_size,
                                                   self.config.gamma)
        scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen,
                                                  self.config.step_size,
                                                  self.config.gamma)
        sys.stdout.flush()

        for epoch in range(self.config.train_epochs):
            scheduler_disc.step()
            scheduler_gen.step()

            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))

            postfix = self._train_epoch(
                epoch, model, tqdm_data,
                optimizer_disc, 
                optimizer_gen)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None and epoch%100==0 and epoch!=0:
                tqdm_data = val_loader
                postfix = self._train_epoch(epoch, model, tqdm_data)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)

            sys.stdout.flush()
            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                model = model.to('cpu')
                torch.save(
                    model.state_dict(),
                    self.config.model_save + f'/ckpt_{self.config.target}_{epoch}.pt'
                )
                model = model.to(device)

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_collate_fn(self, model):
# =============================================================================
#         device = self.get_collate_device(model)
# 
#         def collate(data):
#             tensors = torch.tensor([t for t in data],
#                                    dtype=torch.float64, device=device)
#             return tensors
# =============================================================================
        def collate(data):
            return data
        return collate

    def _get_dataset_info(self, data, name=None):
        df = pd.DataFrame(data)
        maxlen = df.iloc[:, 0].map(len).max()
        ctr = Counter(''.join(df.unstack().values))
        charset = ''
        for c in list(ctr):
            charset += c
        return {"maxlen": maxlen, "charset": charset, "name": name}

    def fit(self,
            model,
            content_train,
            style_instance,
            content_test=None):
        self.MLP_s1 = model.MLP_s1
        self.MLP_s2 = model.MLP_s2
        self.Generator = model.Generator
        self.discriminator = model.Discriminator
        self.heteroencoder = model.heteroencoder
        self.IAF_s1 = model.IAF_s1
        self.IAF_s2 = model.IAF_s2
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            print('Using CUDA')
            self.discriminator.cuda()
            self.MLP_s1.cuda()
            self.MLP_s2.cuda()
            self.Generator.cuda()
            self.heteroencoder.cuda()
            self.IAF_s1.cuda()
            self.IAF_s2.cuda()
        else:
            print('Using CPU')

        logger = Logger() if self.config.log_file is not None else None
        

        _, smi2vec = load_model()
        print("Training GAN.")
        def get_latent(smiles):
            vec = smi2vec(smiles)
            latent = self.heteroencoder.encode(vec)
            latent = latent.reshape(latent.shape[0], self.latent_size)
            return latent
        os.makedirs('./latent_save', exist_ok=True)
        if not os.path.exists(f'./latent_save/latent_content_train_{self.config.target}.pkl'):
            latent_content_train = get_latent(content_train)
            latent_style_instance = get_latent(style_instance)
            
            pickle.dump(
                latent_content_train, 
                open(f'./latent_save/latent_content_train_{self.config.target}.pkl', 'wb'))
            pickle.dump(
                latent_style_instance, 
                open(f'./latent_save/latent_style_instance_train_{self.config.target}.pkl', 'wb'))
            if content_test is not None:
                latent_content_test = get_latent(content_test)
                pickle.dump(
                    latent_content_test, 
                    open(f'./latent_save/latent_content_test_{self.config.target}.pkl', 'wb'))
        else:
            latent_content_train = pickle.load(
                open(f'./latent_save/latent_content_train_{self.config.target}.pkl', 'rb'))
            latent_style_instance = pickle.load(
                open(f'./latent_save/latent_style_instance_train_{self.config.target}.pkl', 'rb'))
            if content_test is not None:
                latent_content_test = pickle.load(
                    open(f'./latent_save/latent_content_test_{self.config.target}.pkl', 'rb'))
        

        train_loader = self.get_dataloader(
                model,
                MolStyDataset(latent_content_train, latent_style_instance, is_train=True),
                shuffle=True, drop_last=True)
        val_loader = None if content_test is None else self.get_dataloader(
            model, MolStyDataset(latent_content_test, latent_style_instance), shuffle=False
        , drop_last=True)

        self._train(model, train_loader, val_loader, logger)
        return model





import warnings
warnings.filterwarnings('ignore')

from rdkit.Chem import Descriptors
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

        return prop_array

    except:
        return None
        # return [0, 0, 0, 0, 0, 0]

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
# =============================================================================
#         # SA score
#         props.append(sascorer.calculateScore(mol))
# =============================================================================
        
        return props

    except:
        return [0]*26 # [0]*27

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
        closest_sims.append(np.nanmax(active_sims))
        closest_sims_id.append(np.argmax(active_sims))

    return np.array(closest_sims), np.array(closest_sims_id)

from collections import defaultdict
def lads_score_v2(actives_mol, decoys_mol):
    # Similar to DEKOIS (v2)
    # Lower is better (less like actives), higher is worse (more like actives)
    active_fps = []
    active_info = {}
    info={}
    atoms_per_bit = defaultdict(int)
    for m in actives_mol:
        # m = Chem.MolFromSmiles(smi)
        active_fps.append(AllChem.GetMorganFingerprint(m,3,useFeatures=True, bitInfo=info))
        for key in info:
            if key not in active_info:
                active_info[key] = info[key]
                env = Chem.FindAtomEnvironmentOfRadiusN(m, info[key][0][1], info[key][0][0])
                amap={}
                submol=Chem.PathToSubmol(m,env,atomMap=amap)
                if info[key][0][1] == 0:
                    atoms_per_bit[key] = 1
                else:
                    atoms_per_bit[key] = submol.GetNumHeavyAtoms()

    decoys_fps = [AllChem.GetMorganFingerprint(mol,3,useFeatures=True) for mol in decoys_mol] # Roughly FCFP_6

    master_active_fp_freq = defaultdict(int)
    for fp in active_fps:
        fp_dict = fp.GetNonzeroElements()
        for k, v in fp_dict.items():
            master_active_fp_freq[k] += 1
    # Reweight
    for k in master_active_fp_freq:
        # Normalise
        master_active_fp_freq[k] /= len(active_fps)
        # Weight by size of bit
        master_active_fp_freq[k] *= atoms_per_bit[k]

    decoys_lads_avoid_scores = [sum([master_active_fp_freq[k] for k in decoy_fp.GetNonzeroElements()])/len(decoy_fp.GetNonzeroElements()) 
                                for decoy_fp in decoys_fps]
    
    return decoys_lads_avoid_scores