import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.train import make_predictions
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = parse_train_args()
    # args.checkpoint_dir = './ckpt'
    modify_train_args(args)
    
    if args.data_path.endswith('csv'):
        df = pd.read_csv(args.data_path, sep=' ', header=None, names=['act', 'smiles'])[['smiles']]
    else:
        df = pd.read_csv(args.data_path, header=None, names=['act', 'smiles'], sep=' ')[['smiles']]
    pred, smiles = make_predictions(args, df.smiles.tolist())
    df = pd.DataFrame({'smiles':smiles})
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
    
    df = df[['smiles', 'pred_0']]
    df.to_csv('./predict_TOX.csv', index=False)
    
    plt.figure(figsize=(16,9), dpi=300)
    sns.distplot(df['pred_0'])
    plt.savefig('./fig.jpg')