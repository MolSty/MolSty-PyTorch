import argparse
import os
import sys
import torch
import rdkit
import pickle
import pandas as pd
import numpy as np

from moses.script_utils import add_train_args, read_smiles_csv, set_seed
from moses.models_storage import ModelsStorage
from moses.dataset import get_dataset

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )
    return parser


def main(model, config):
    os.makedirs(config.model_save, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    content_train = pickle.load(open(f'./data/content_train_{config.target}.pkl', 'rb'))
    style_instance = pickle.load(open(f'./data/style_instance_train_{config.target}.pkl', 'rb'))
    trainer = MODELS.get_model_trainer(model)(config)

    vocab = None
    model = MODELS.get_model_class(model)(vocab, config)
    if config.model_load is not None:
        print(f'load model from {config.model_load}')
        model_state = torch.load(config.model_load)
        model.load_state_dict(model_state)
    model = model.to(device)
    trainer.fit(model, 
                content_train=content_train, 
                style_instance=style_instance)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
