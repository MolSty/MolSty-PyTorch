import argparse
import sys
import torch
import rdkit
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed
from moses.latentgan.model import load_model
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import os
from tqdm import trange

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()
import pickle

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models sampler script', description='available models')
    for model in MODELS.get_model_names():
        add_sample_args(subparsers.add_parser(model))
    return parser


def main(model, config):
    set_seed(2048)

    model_config = torch.load('./temp')
    model_state = torch.load(config.model_load)
    model_vocab = None
    model = MODELS.get_model_class(model)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.cuda()
    model.eval()
    
    model.model_loaded = True
    _, smi2vec = load_model()

    content_test = pickle.load(
        open(f'./data/content_test_{config.target}.pkl', 'rb'))[:100]
    style_instance = pickle.load(open(f'./data/style_instance_test_{config.target}.pkl', 'rb'))# [:30000]
    
    latent_content_test = model.heteroencoder.encode(smi2vec(content_test))
    latent_style_instance = model.heteroencoder.encode(smi2vec(style_instance))
    print(latent_content_test.shape, latent_style_instance.shape)
    samples = model.sample_per_act(
        latent_content_test, latent_style_instance, num_per_act=10,
        n_ins=config.n_ins)
    os.makedirs('./eval/results', exist_ok=True)
    pickle.dump(samples, open(f'./eval/results/{config.target}_result.pkl', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_load')
    parser.add_argument('--target', default='SA')
    parser.add_argument('--n_ins', default=10, type=int, )
    config = parser.parse_args()
    model = 'MolSty'
    main(model, config)
