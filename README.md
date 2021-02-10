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

`python ./scripts/train.py MolSty --target TARGET --model_save ./checkpoints`

where `<TARGET>` is the name of dataset, "SA" and "TOX" are supported in this repository.

To generate molecules using MolSty, use:

`sh generate_SA.sh PATH_CHECKPOINT` and `sh generate_TOX.sh PATH_CHECKPOINT`

where, `<PATH_CHECKPOINT>` is the checkpoint path, for example, `./checkpoints/ckpt_SA_100.pt`

To evaluate the generated molecules, please unzip file ROOT/eval/zinc_all_alerts_pred.zip first, and use:

`python ./eval/eval.py --target TARGET`
