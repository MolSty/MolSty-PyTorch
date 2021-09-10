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

class MolSty(nn.Module):
    def __init__(self, vocabulary, config):
        super(MolSty, self).__init__()
        self.vocabulary = vocabulary
        h = config.latent_vector_dim
        z = h
        self.MLP_s1 = MLP(h, z)
        self.MLP_s2 = MLP(h, z)
        self.Generator = MLP(z*2, h)
        self.model_version = config.heteroencoder_version
        self.Discriminator = Discriminator(
            data_shape=(1, h), output_shape=3)
        self.IAF_s1 = IAF(n_z=h, n_h=h, n_made=h, flow_depth=6)
        self.IAF_s2 = IAF(n_z=h, n_h=h, n_made=h, flow_depth=6)
        self.heteroencoder, _ = load_model()
        self.model_loaded = False
        # init params
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.Discriminator.cuda()
            self.MLP_s1.cuda()
            self.MLP_s2.cuda()
            self.Generator.cuda()
            self.IAF_s1.cuda()
            self.IAF_s2.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        print(self.Generator)
        print(self.IAF_s1)

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

    def sample_per_act(self, content, style, num_per_act=10, n_ins=10):
        self._log2pi = torch.log(torch.tensor(
                2*np.pi, dtype=torch.float, device=self.device))
        content = torch.Tensor(content).cuda()
        style = torch.Tensor(style).cuda()
        smi_lst = [[] for i in range(content.shape[0])]
        hidden_size = content.shape[-1]
        for i in trange(content.shape[0]):
            if len(smi_lst[i])==num_per_act:
                continue
            while True:
                c = content[i:i+1].clone()
                c = c.repeat(num_per_act*5, 1)
                style_ins = style[random.sample(list(range(style.shape[0])),
                                                c.shape[0]*n_ins)]
                s, _ = self.extract_style_features(
                    style_ins.view(n_ins, -1, hidden_size).contiguous(), 
                    self.IAF_s2)
                latent = self.Generator(torch.cat([
                    c, self.MLP_s2(s)], -1)).detach()
                gens_smiles = self.heteroencoder.decode(latent)
                for j in range(len(gens_smiles)):
                    smi = gens_smiles[j].replace('\n', '')
                    if len(smi_lst[i])<num_per_act and Chem.MolFromSmiles(smi) is not None:
                        smi_lst[i].append(smi)
                if len(smi_lst[i]) == num_per_act:
                    break
        return np.array(smi_lst)
    
    def build_initial_gaussian(self, instances):
        # instance: (K, B, T)
        assert len(instances.shape) == 3
        n_ins = instances.size(0)
# =============================================================================
#         bs = instances.size(1)
#         hidden_size = instances.size(2)
# =============================================================================
        
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

        # log q(z_t|c) = log q(z_0|c) - sum log det|d_zt/dz_(t-1)|
        # log q(z) = -0.5*log(2*pi) - log(sigma) - 0.5 * eps**2
        log_qz0 = - 0.5*self._log2pi - 0.5 * torch.log(std_sq+1e-10) - 0.5 * eps**2

        # build flow
        z, log_det = IAF(z0, h)

        log_qz = log_qz0 + log_det

        return z, log_qz

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


class MolStyDataset(data.Dataset):
    def __init__(self, content, style, is_train=False, n_ins=10):
        self.content = content
        self.style = style
        self.is_train = is_train
        self.n_ins = n_ins

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        content = self.content[index]
        style_ins = self.style[np.random.randint(len(self.style))]
        return content, style_ins


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


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

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


from torch import nn
import torch.nn.functional as F

class InverseAutoregressiveBlock(nn.Module):
    """The Inverse Autoregressive Flow block,
    https://arxiv.org/abs/1606.04934"""
    def __init__(self, n_z, n_h, n_made):
        super(InverseAutoregressiveBlock, self).__init__()

        # made: take as inputs: z_{t-1}, h; output: m_t, s_t
        self.made = MADE(num_input=n_z, num_output=n_z * 2,
                     num_hidden=n_made, num_context=n_h)
        self.sigmoid_arg_bias = nn.Parameter(torch.ones(n_z) * 2)


    def forward(self, prev_z, h):
        '''
        prev_z: z_{t-1}
        h: the context
        '''
        m, s = torch.chunk(self.made(prev_z, h), chunks=2, dim=-1)
        # the bias is used to make s sufficiently positive
        #   see Sec. 4 in (Kingma et al., 2016) for more details
        s = s + self.sigmoid_arg_bias
        sigma = torch.sigmoid(s)
        z = sigma * prev_z + (1 - sigma) * m

        log_det = -F.logsigmoid(s)

        return z, log_det

class IAF(nn.Module):
    """docstring for IAF"""
    def __init__(self, n_z, n_h, n_made, flow_depth):
        super(IAF, self).__init__()
        self._flow_depth = flow_depth
        self._flows = nn.ModuleList(
            [InverseAutoregressiveBlock(n_z, n_h, n_made)
             for _ in range(0, flow_depth)])

        self._reverse_idxes = np.array(np.arange(0, n_z)[::-1])

    def _do_reverse(self, v):
        return v[:, self._reverse_idxes]

    def forward(self, z, h):
        total_log_det = torch.zeros_like(z, device=z.device)
        for i, flow in enumerate(self._flows):
            z, log_det = flow(z, h)
            z = self._do_reverse(z)
            total_log_det += log_det
        return z, total_log_det



# The implemention of MADE: Masked Autoencoder for Distribution Estimation (https://arxiv.org/abs/1502.03509)
#   which is directly borrowed from https://github.com/altosaar/variational-autoencoder/blob/master/flow.py

class MaskedLinear(nn.Module):
  """Linear layer with some input-output connections masked."""
  def __init__(self, in_features, out_features, mask, context_features=None, bias=True):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features, bias)
    self.register_buffer("mask", mask)
    if context_features is not None:
      self.cond_linear = nn.Linear(context_features, out_features, bias=False)

  def forward(self, input, context=None):
    output =  F.linear(input, self.mask * self.linear.weight, self.linear.bias)
    if context is None:
      return output
    else:
      return output + self.cond_linear(context)


class MADE(nn.Module):
  def __init__(self, num_input, num_output, num_hidden, num_context):
    super().__init__()
    # m corresponds to m(k), the maximum degree of a node in the MADE paper
    self._m = []
    self._masks = []
    self._build_masks(num_input, num_output, num_hidden, num_layers=3)
    self._check_masks()
    modules = []
    self.input_context_net = MaskedLinear(num_input, num_hidden, self._masks[0], num_context)
    modules.append(nn.ReLU())
    modules.append(MaskedLinear(num_hidden, num_hidden, self._masks[1], context_features=None))
    modules.append(nn.ReLU())
    modules.append(MaskedLinear(num_hidden, num_output, self._masks[2], context_features=None))
    self.net = nn.Sequential(*modules)


  def _build_masks(self, num_input, num_output, num_hidden, num_layers):
    """Build the masks according to Eq 12 and 13 in the MADE paper."""
    rng = np.random.RandomState(0)
    # assign input units a number between 1 and D
    self._m.append(np.arange(1, num_input + 1))
    for i in range(1, num_layers + 1):
      # randomly assign maximum number of input nodes to connect to
      if i == num_layers:
        # assign output layer units a number between 1 and D
        m = np.arange(1, num_input + 1)
        assert num_output % num_input == 0, "num_output must be multiple of num_input"
        self._m.append(np.hstack([m for _ in range(num_output // num_input)]))
      else:
        # assign hidden layer units a number between 1 and D-1
        self._m.append(rng.randint(1, num_input, size=num_hidden))
        #self._m.append(np.arange(1, num_hidden + 1) % (num_input - 1) + 1)
      if i == num_layers:
        mask = self._m[i][None, :] > self._m[i - 1][:, None]
      else:
        # input to hidden & hidden to hidden
        mask = self._m[i][None, :] >= self._m[i - 1][:, None]
      # need to transpose for torch linear layer, shape (num_output, num_input)
      self._masks.append(torch.from_numpy(mask.astype(np.float32).T))

  def _check_masks(self):
    """Check that the connectivity matrix between layers is lower triangular."""
    # (num_input, num_hidden)
    prev = self._masks[0].t()
    for i in range(1, len(self._masks)):
      # num_hidden is second axis
      prev = prev @ self._masks[i].t()
    final = prev.numpy()
    num_input = self._masks[0].shape[1]
    num_output = self._masks[-1].shape[0]
    assert final.shape == (num_input, num_output)
    if num_output == num_input:
      assert np.triu(final).all() == 0
    else:
      for submat in np.split(final,
            indices_or_sections=num_output // num_input,axis=1):
        assert np.triu(submat).all() == 0


  def forward(self, input, context=None):
    # first hidden layer receives input and context
    hidden = self.input_context_net(input, context)
    # rest of the network is conditioned on both input and context
    return self.net(hidden)