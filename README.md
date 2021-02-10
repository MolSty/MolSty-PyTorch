# Molecular Attributes Transfer from Non-Parallel Data
The code was built based on [MOSES](https://github.com/molecularsets/moses). Thanks a lot for their code sharing!

# Requirements
+ Python 3.6
+ PyTorch 1.6
+ CUDA 10.2
+ RDKit

# Quick start
## Installation
`python setup.py install`

## Dataset
We have prepared two training datasets based on different properties, toxicity and SA (Synthetic Accessibility). Both were created from a subset of the ZINC dataset.

## Running MolSty
To train model using MolSty, use:

`python ./scripts/train.py MolSty --target SA --model_save ./checkpoints`

To generate molecules using MolSty, use:

`sh generate.sh PATH_CHECKPOINT`

where, `PATH_CHECKPOINT` ./checkpoints/ckpt_SA_100.pt
