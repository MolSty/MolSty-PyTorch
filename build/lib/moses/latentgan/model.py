import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd as autograd
from rdkit import Chem
import pickle
from tqdm import trange
import random

class LatentGAN(nn.Module):
    def __init__(self, vocabulary, config):
        super(LatentGAN, self).__init__()
        self.vocabulary = vocabulary
        h = config.latent_vector_dim
        z = h
        self.encoder_c = Generator(h, z)
        self.encoder_s1 = Generator(h, z)
        self.encoder_s2 = Generator(h, z)
        self.decoder = Generator(z*2, h)
        self.model_version = config.heteroencoder_version
        self.Discriminator = Discriminator(
            data_shape=(1, h), output_shape=3)
        
        self.heteroencoder, _ = load_model()
        self.model_loaded = False
        # init params
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.Discriminator.cuda()
            self.encoder_c.cuda()
            self.encoder_s1.cuda()
            self.encoder_s2.cuda()
            self.decoder.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        print(self.encoder_c)
        print(self.decoder)

    def forward(self, n_batch):
        out = self.sample(n_batch)
        return out

    def encode_smiles(self, smiles_in, encoder=None):

        model = load_model(model_version=encoder)

        # MUST convert SMILES to binary mols for the model to accept them
        # (it re-converts them to SMILES internally)
        mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles))
                   for smiles in smiles_in]
        latent = model.transform(model.vectorize(mols_in))

        return latent.tolist()

    def compute_gradient_penalty(self, real_samples,
                                 fake_samples, discriminator):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples +
                        ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)# [:, 2].view(-1, 1)
        fake = self.Tensor(real_samples.shape[0], 3).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_per_act(self, content, style, num_per_act=500):
        self.S_c = Sampler(generator=self.encoder_c)
        self.S_s2 = Sampler(generator=self.encoder_s2)
        content = torch.Tensor(content).cuda()
        style = torch.Tensor(style).cuda()
        smi_lst = [[] for i in range(content.shape[0])]
        for i in trange(content.shape[0]):
            if len(smi_lst[i])==num_per_act:
                continue
            while True:
                c = content[i:i+1].clone()
                c = c.repeat(num_per_act*5, 1)
                s = style[random.sample(list(range(style.shape[0])),c.shape[0])]
                latent = self.decoder(torch.cat([
                    self.S_c.sample(c), 
                    self.S_s2.sample(s)], -1)).detach()
                gens_smiles = self.heteroencoder.decode(latent)
                for j in range(len(gens_smiles)):
                    smi = gens_smiles[j].replace('\n', '')
                    if len(smi_lst[i])<num_per_act and Chem.MolFromSmiles(smi) is not None:
                        smi_lst[i].append(smi)
                if len(smi_lst[i]) == num_per_act:
                    break
        return np.array(smi_lst)
    

from moses.vae.trainer import VAETrainer
from moses.vae.model import VAE

class temp_cfg():
    def __init__(self):
        pass
    @property
    def freeze_embeddings(self):
        return False
    @property
    def q_cell(self):
        return 'gru'
    @property
    def q_d_h(self):
        return 256
    @property
    def q_n_layers(self):
        return 1
    @property
    def q_bidir(self):
        return False
    @property
    def d_cell(self):
        return 'gru'
    @property
    def d_n_layers(self):
        return 3
    @property
    def d_dropout(self):
        return 0
    @property
    def d_z(self):
        return 128
    @property
    def d_d_h(self):
        return 512
    @property
    def n_batch(self):
        return 512
    @property
    def clip_grad(self):
        return 50
    @property
    def kl_start(self):
        return 0
    @property
    def kl_w_start(self):
        return 0
    @property
    def kl_w_end(self):
        return 0.05
    @property
    def lr_start(self):
        return 3 * 1e-4
    @property
    def lr_n_period(self):
        return 10
    @property
    def lr_n_restarts(self):
        return 10
    @property
    def lr_n_mult(self):
        return 1
    @property
    def lr_end(self):
        return 3 * 1e-4
    @property
    def n_last(self):
        return 1000
    @property
    def n_jobs(self):
        return 1
    @property
    def n_workers(self):
        return 1

def load_model(model_version=None):
    config = temp_cfg()
    trainer = VAETrainer(config)
    vocab_load = './vocab_vae'
    vocab = torch.load(vocab_load)

    model = VAE(vocab, config)#.to(device)
    
    model.load_state_dict(torch.load('./vae.pt'))
    try:
        model.cuda()
    except:
        pass
    model.eval()
    
    collate_fn = trainer.get_collate_fn(model)
    return model, collate_fn


class LatentMolsDataset(data.Dataset):
    def __init__(self, act, dec, is_train=False):
        self.act = act
        self.dec = dec
        self.is_train = is_train

    def __len__(self):
        return len(self.act)

    def __getitem__(self, index):
        return self.act[index], self.dec[np.random.randint(len(self.dec))]


class Discriminator(nn.Module):
    def __init__(self, data_shape=(1, 512), output_shape=1):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_shape),
        )

    def forward(self, mol):
        pred = self.model(mol)
        return pred


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()

        # latent dim of the generator is one of the hyperparams.
        # by default it is set to the prod of data_shapes
        self.in_dim = in_dim
        self.out_dim = out_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.gru = nn.GRU(128, 128)
        self.model = nn.Sequential(
            *block(self.in_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.out_dim),
            # nn.Tanh() # expecting latent vectors to be not normalized
        )

    def forward(self, z):
        out = self.model(z)
        return out#, new_z


class Sampler(object):
    """
    Sampling the mols the generator.
    All scripts should use this class for sampling.
    """

    def __init__(self, generator: Generator):
        self.G = generator

    def sample(self, z=None, ref=None):
        if z is None:
            n = ref.shape[0]
            latent_dim = ref.shape[1]
            # Sample noise as generator input
            z = torch.cuda.FloatTensor(
                    np.random.uniform(-1, 1, (n, latent_dim)))
        # Generate a batch of mols
        return self.G(z)
