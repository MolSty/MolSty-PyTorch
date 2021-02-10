import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['fn', 'Dataset', 'Orig num actives', 'Num actives', 
        'Num generated mols', 'Num unique gen mols',
        'AUC ROC - 1NN', 'AUC', 'DOE',
        'LADS', 'DG', 'Doppelganger score max']
df1 = pd.read_csv('./results.csv', header=None, names=cols)
df2 = pd.read_csv('./results_deepcoy.csv', header=None, names=cols)
df3 = pd.read_csv('../../data/results_dude.csv', header=None, 
                  names=['fn', 'Dataset', 
                         'AUC ROC - 1NN', 'AUC', 'DOE',
                         'LADS', 'DG', 'Doppelganger score max'])
df3 = df3.drop_duplicates('fn').reset_index(drop=True)
df1['fn'] = df1['fn'].apply(lambda x: x.split('_')[0])
df2['fn'] = df2['fn'].apply(lambda x: x.split('-')[2].upper())

df1 = df1.sort_values(by=['fn']).reset_index(drop=True)
df2 = df2.sort_values(by=['fn']).reset_index(drop=True)
df3 = df3.sort_values(by=['fn']).reset_index(drop=True)

s = set(df1.fn)
df2 = df2.loc[df2.fn.isin(s)]
df3 = df3.loc[df3.fn.isin(s)]

df = df1[['fn', 'AUC', 'DOE', 'LADS', 'DG']].rename(
    columns={'AUC':'AUC_styleGAN', 'DOE':'DOE_styleGAN', 
             'LADS':'LADS_styleGAN', 'DG':'DG_styleGAN'}).merge(
     df2[['fn', 'AUC', 'DOE', 'LADS', 'DG']].rename(
    columns={'AUC':'AUC_deepcoy', 'DOE':'DOE_deepcoy', 
             'LADS':'LADS_deepcoy', 'DG':'DG_deepcoy', }), 
     on='fn', how='inner').merge(
     df3[['fn', 'AUC', 'DOE', 'LADS', 'DG']].rename(
    columns={'AUC':'AUC_DUDE', 'DOE':'DOE_DUDE', 
             'LADS':'LADS_DUDE', 'DG':'DG_DUDE', }), 
     on='fn', how='inner')
# =============================================================================
# df = df.loc[df.LADS_DUDE>df.LADS_deepcoy]
# =============================================================================
# =============================================================================
# df = df.loc[df.DOE_styleGAN<0.2].reset_index(drop=True)
# =============================================================================
plt.figure(figsize=(9,6), dpi=300)
sns.boxplot(data=df[['AUC_DUDE', 'AUC_deepcoy', 'AUC_styleGAN']])
plt.show()

plt.figure(figsize=(9,6), dpi=300)
plt.plot(df['fn'],df['DOE_DUDE'],'o-',label=f"DOE_DUDE (mean={df.DOE_DUDE.mean():.3f})")
plt.plot(df['fn'],df['DOE_deepcoy'],'o-',label=f"DOE_DeepCoy (mean={df.DOE_deepcoy.mean():.3f})")
plt.plot(df['fn'],df['DOE_styleGAN'],'o-',label=f"DOE_StyleGAN (mean={df.DOE_styleGAN.mean():.3f})")
plt.xticks(rotation=45)
plt.legend()
plt.show()

plt.figure(figsize=(9,6), dpi=300)
plt.plot(df['fn'],df['LADS_DUDE'],'o-',label=f"LADS_DUDE (mean={df.LADS_DUDE.mean():.3f})")
plt.plot(df['fn'],df['LADS_deepcoy'],'o-',label=f"LADS_DeepCoy (mean={df.LADS_deepcoy.mean():.3f})")
plt.plot(df['fn'],df['LADS_styleGAN'],'o-',label=f"LADS_StyleGAN (mean={df.LADS_styleGAN.mean():.3f})")
plt.xticks(rotation=45)
plt.legend()
plt.show()

plt.figure(figsize=(9,6), dpi=300)
plt.plot(df['fn'],df['DG_DUDE'],'o-',label=f"DG_DUDE (mean={df.DG_DUDE.mean():.3f})")
plt.plot(df['fn'],df['DG_deepcoy'],'o-',label=f"DG_DeepCoy (mean={df.DG_deepcoy.mean():.3f})")
plt.plot(df['fn'],df['DG_styleGAN'],'o-',label=f"DG_StyleGAN (mean={df.DG_styleGAN.mean():.3f})")
plt.xticks(rotation=45)
plt.legend()
plt.show()