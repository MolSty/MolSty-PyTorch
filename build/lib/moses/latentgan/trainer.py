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
from .model import LatentMolsDataset
from .model import load_model
from .model import Sampler
import pickle

class LatentGANTrainer(MosesTrainer):

    def __init__(self, config):
        self.config = config
        self.latent_size = self.config.latent_vector_dim
# =============================================================================
#         acts_smiles, decs_smiles = pickle.load(
#             open(f'./data/test_{self.config.target.upper()}.pkl', 'rb'))
# =============================================================================
# =============================================================================
#         acts_smiles = pickle.load(
#             open(f'./data/content_test_{self.config.target.upper()}.pkl', 'rb'))
#         decs_smiles = pickle.load(
#             open(f'./data/style_instance_{self.config.target.upper()}.pkl', 'rb'))
#         self.acts_mols = [Chem.MolFromSmiles(smiles) for smiles in set(acts_smiles)]
#         self.decs_mols = [Chem.MolFromSmiles(smiles) for smiles in set(decs_smiles)]
#         self.active_props = [calc_props_dude(mol) for mol in self.acts_mols]
#         self.decoy_props = [calc_props_dude(mol) for mol in self.decs_mols]
# =============================================================================

    def _train_epoch(self, epoch, model, tqdm_data,
                     optimizer_disc=None, 
                     optimizer_gen=None):
        if optimizer_disc is None:
            model.eval()
            optimizer_gen = None
        else:
            model.train()
        self.Sampler_c = Sampler(generator=self.encoder_c)
        self.Sampler_s1 = Sampler(generator=self.encoder_s1)
        self.Sampler_s2 = Sampler(generator=self.encoder_s2)

        postfix = {}
        disc_loss_batch = []
        g_loss_batch = []
        L_FP_content_batch = []
        L_FP_style_batch = []
        L_FPT_style_batch = []
        L_FPD_batch = []
        L_recon_batch = []
        L_recon2_batch = []
        L_cycle_batch = []

        for i, item in enumerate(tqdm_data): # real_mols
            act_mols, dec_mols = item[:, 0, :], item[:, 1, :]
            mc = act_mols.type(model.Tensor)
            ms = dec_mols.type(model.Tensor)
            
            if optimizer_disc is not None:
                optimizer_disc.zero_grad()
            
            cc = self.Sampler_c.sample(mc)
            sc = self.Sampler_s1.sample(mc)
            cs = self.Sampler_c.sample(ms)
            ss = self.Sampler_s2.sample(ms)
            
            # 生成新的分子style&content排列组合形式
            ccsc = self.decoder(torch.cat([cc, sc], -1))
            ccss = self.decoder(torch.cat([cc, ss], -1))
            cssc = self.decoder(torch.cat([cs, sc], -1))
            csss = self.decoder(torch.cat([cs, ss], -1))
            
            style_1_validity = F.log_softmax(
                self.discriminator(torch.cat([mc, ccsc], 0)), dim=1)[:, 0]
            style_2_validity = F.log_softmax(
                self.discriminator(torch.cat([ms, csss], 0)), dim=1)[:, 1]
            fake_validity = F.log_softmax(
                self.discriminator(torch.cat([ccss, cssc], 0)), dim=1)[:, 2]
            
            # Gradient penalty
            gradient_penalty = model.compute_gradient_penalty(
                torch.cat([mc, ccsc], 0).data, 
                torch.cat([ccss, cssc], 0).data, self.discriminator)
            
            d_loss = - torch.mean(style_1_validity) - torch.mean(style_2_validity)\
                - torch.mean(fake_validity) + self.config.gp * gradient_penalty
                
            disc_loss_batch.append(d_loss.item())

            if optimizer_disc is not None:
                
                d_loss.backward()
                optimizer_disc.step()
                
                # Train the generator every n_critic steps
                if i % self.config.n_critic == 0:
                    def triplet_loss(anchor, positive, negative, margin=0.2):
                    
                        d_pos = torch.sum((anchor - positive)**2, 1)
                        d_neg = torch.sum((anchor - negative)**2, 1)
                        tmp = torch.cat([(margin + d_pos - d_neg).unsqueeze(-1), 
                                         torch.zeros_like(d_pos.unsqueeze(-1)).cuda()], -1)
                        loss, _ = torch.max(tmp, -1)
                        return torch.mean(loss)#**2
                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_gen.zero_grad()
                    
                    cc = self.Sampler_c.sample(mc)
                    sc = self.Sampler_s1.sample(mc)
                    cs = self.Sampler_c.sample(ms)
                    ss = self.Sampler_s2.sample(ms)
                    
                    # 生成新的分子style&content排列组合形式
                    ccsc = self.decoder(torch.cat([cc, sc], -1))
                    ccss = self.decoder(torch.cat([cc, ss], -1))
                    cssc = self.decoder(torch.cat([cs, sc], -1))
                    csss = self.decoder(torch.cat([cs, ss], -1))
                    
                    style_1_validity = F.log_softmax(
                        self.discriminator(torch.cat([mc, ccsc], 0)), dim=1)[:, 0]
                    style_2_validity = F.log_softmax(
                        self.discriminator(torch.cat([ms, csss], 0)), dim=1)[:, 1]
                    fake_validity = F.log_softmax(
                        self.discriminator(torch.cat([ccss, cssc], 0)), dim=1)[:, 2]
            
                    # Gradient penalty
                    gradient_penalty = model.compute_gradient_penalty(
                        torch.cat([mc, ccsc], 0).data, 
                        torch.cat([ccss, cssc], 0).data, self.discriminator)
                    
                    g_loss = - torch.mean(style_1_validity) - torch.mean(style_2_validity)\
                        - torch.mean(fake_validity) + self.config.gp * gradient_penalty
                    
                    ccsc_decode_s = self.Sampler_s1.sample(ccsc) # mc
                    ccss_decode_s = self.Sampler_s2.sample(ccss)
                    cssc_decode_s = self.Sampler_s1.sample(cssc)
                    csss_decode_s = self.Sampler_s2.sample(csss) # ms
# =============================================================================
#                     
# =============================================================================
                    
                    L_FP_style =  0.5*triplet_loss(
                        ss, ccss_decode_s, sc) # style(ms) - style(ccss) - style(mc)
                    L_FP_style += 0.5*triplet_loss(
                        sc, cssc_decode_s, ss)
                    g_loss += L_FP_style  * 10
                    

                    L_recon = 0.5 *  F.mse_loss(
                        self.decoder(torch.cat([cc, sc], -1)),
                        mc)
                    L_recon += 0.5 * F.mse_loss(
                        self.decoder(torch.cat([cs, ss], -1)),
                        ms)
                    g_loss += L_recon  * 50
                    
                    L_recon2 = 0.5 * F.mse_loss(
                        self.decoder(torch.cat([cc, ss], -1)),
                        mc)
                    L_recon2 +=0.5 * F.mse_loss(
                        self.decoder(torch.cat([cs, sc], -1)),
                        ms)
                    g_loss += L_recon2  * 10
                    
                    L_cycle =  0.5 * F.mse_loss(
                        ccsc_decode_s,
                        sc)
                    L_cycle += 0.5 * F.mse_loss(
                        csss_decode_s,
                        ss)
                    g_loss += L_cycle * 10
                    
# =============================================================================
#                     L_cycle_content =  0.5 * F.mse_loss(
#                         self.Sampler_c.sample(ccsc),
#                         cc)
#                     L_cycle_content += 0.5 * F.mse_loss(
#                         self.Sampler_c.sample(csss),
#                         sc)
#                     g_loss += L_cycle_content * 10
# =============================================================================
# =============================================================================
#                     L_FPD = 0.5*triplet_loss(
#                         c1s1_decode_s, s1_style, c2s1_decode_s, margin=0.)
#                     L_FPD += 0.5*triplet_loss(
#                         c2s2_decode_s, s2_style, c1s2_decode_s, margin=0.)
#                     g_loss += L_FPD * 10
# =============================================================================

                    g_loss.backward()
                    optimizer_gen.step()
                    g_loss_batch.append(g_loss.item())
                    postfix['g_loss'] = np.mean(g_loss_batch)
# =============================================================================
#                     L_FP_content_batch.append(L_FP_content.item())
#                     postfix['L_FP_content'] = np.mean(L_FP_content_batch)
# =============================================================================
                    
                    L_FP_style_batch.append(L_FP_style.item())
                    postfix['L_FP_style'] = np.mean(L_FP_style_batch)
# =============================================================================
#                     L_FPT_style_batch.append(L_FPT_style.item())
#                     postfix['L_FPT_style'] = np.mean(L_FPT_style_batch)
# =============================================================================
# =============================================================================
#                     L_FPD_batch.append(L_FPD.item())
#                     postfix['L_FPD'] = np.mean(L_FPD_batch)
# =============================================================================
                    L_recon_batch.append(L_recon.item())
                    postfix['L_recon'] = np.mean(L_recon_batch)
                    L_recon2_batch.append(L_recon2.item())
                    postfix['L_recon2'] = np.mean(L_recon2_batch)
                    
                    L_cycle_batch.append(L_cycle.item())
                    postfix['L_cycle'] = np.mean(L_cycle_batch)
                    
            else:
# =============================================================================
#                 fake_mols = torch.cat([ccss], 0).detach()#.cpu().numpy()
#                 gens_smiles = self.heteroencoder.decode(fake_mols)
#                 
#                 gens_mols = []
#                 for smiles in gens_smiles:
#                     try:
#                         mol = Chem.MolFromSmiles(smiles)
#                         if mol is not None:
#                             gens_mols.append(mol)
#                     except:
#                         pass
#                 if len(gens_mols) < 10:
#                     continue
#                 doe = np.array([doe_score(self.active_props, self.decoy_props)])
#                 dg, _ = dg_score(self.acts_mols, self.decs_mols)
#                 print('raw', round(np.nanmean(doe), 4), round(np.nanmean(dg), 4))
#                 decoy_gen_props = [calc_props_dude(mol) for mol in gens_mols]
#                 decoy_gen_props = [item for item in decoy_gen_props if item is not None]
#                 if len(decoy_gen_props) == 0:
#                     break
#                 doe = np.array([doe_score(self.active_props, decoy_gen_props)])
#                 dg, _ = dg_score(self.acts_mols, gens_mols)
#                 lads = lads_score_v2(self.acts_mols, gens_mols)
#                 print('gen', round(np.nanmean(doe), 4), round(np.nanmean(dg), 4),
#                       round(np.nanmean(lads), 4), 
#                       round(len(decoy_gen_props)/len(gens_smiles), 4))
# =============================================================================
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
                list(self.encoder_c.parameters())+\
                list(self.encoder_s1.parameters())+\
                list(self.encoder_s2.parameters())+\
                list(self.decoder.parameters()),
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
        device = self.get_collate_device(model)

        def collate(data):
            tensors = torch.tensor([t for t in data],
                                   dtype=torch.float64, device=device)
            return tensors

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
        self.encoder_c = model.encoder_c
        self.encoder_s1 = model.encoder_s1
        self.encoder_s2 = model.encoder_s2
        self.decoder = model.decoder
        self.discriminator = model.Discriminator
        self.heteroencoder = model.heteroencoder
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            print('Using CUDA')
            self.discriminator.cuda()
            self.encoder_c.cuda()
            self.encoder_s1.cuda()
            self.encoder_s2.cuda()
            self.decoder.cuda()
            self.heteroencoder.cuda()
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
                open(f'./latent_save/latent_style_instance_{self.config.target}.pkl', 'wb'))
            if content_test is not None:
                latent_content_test = get_latent(content_test)
                pickle.dump(
                    latent_content_test, 
                    open(f'./latent_save/latent_content_test_{self.config.target}.pkl', 'wb'))
        else:
            latent_content_train = pickle.load(
                open(f'./latent_save/latent_content_train_{self.config.target}.pkl', 'rb'))
            latent_style_instance = pickle.load(
                open(f'./latent_save/latent_style_instance_{self.config.target}.pkl', 'rb'))
            if content_test is not None:
                latent_content_test = pickle.load(
                    open(f'./latent_save/latent_content_test_{self.config.target}.pkl', 'rb'))
        

        train_loader = self.get_dataloader(
                model,
                LatentMolsDataset(latent_content_train, latent_style_instance, is_train=True),
                shuffle=True)
        val_loader = None if content_test is None else self.get_dataloader(
            model, LatentMolsDataset(latent_content_test, latent_style_instance), shuffle=False
        )

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