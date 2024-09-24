# weighted ensemble of models
import pandas as pd
import numpy as np

df1 = pd.read_csv('./output.csv')
df2 = pd.read_csv('./output_2.csv')
df3 = pd.read_csv('./output_3.csv')
df4 = pd.read_csv('./output_4.csv')
df5 = pd.read_csv('./output_5.csv')

df_ensemble = df1.copy()

df1.drop(columns=['id'], inplace=True)
df2.drop(columns=['id'], inplace=True)
df3.drop(columns=['id'], inplace=True)
df4.drop(columns=['id'], inplace=True)
df5.drop(columns=['id'], inplace=True)

df_ensemble['target'] = 0.2*df1['target'] + 0.2*df2['target'] + 0.1*df3['target'] + 0.3*df4['target'] + 0.2*df5['target']

df_ensemble.to_csv('submission_ensemble.csv', index=False)

