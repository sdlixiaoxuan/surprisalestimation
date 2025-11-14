import pandas as pd

df = pd.read_excel('EnglishMaterial.xlsx')

df['cut_col'] = df['WORD_ID'].str[0]
df['word_id'] = df['WORD_ID']

dfword = pd.read_excel('MonolingualReadingData.xlsx')
col_to_keep = [
    'WORD_ID',
    'WORD'
]
dfword1 = dfword[col_to_keep].drop_duplicates(subset=['WORD_ID'])
dfword1['word'] = dfword1['WORD']

df_merged = pd.merge(df, dfword1, on='WORD_ID', how='left')

col_to_keep = [
    'cut_col',
    'word',
    'word_id'
]
df = df_merged[col_to_keep]

print(df.head())
print(f"Words number: {df.shape[0]}")
print(f"total cut into: {df['cut_col'].unique()}")

import os
output_path = 'outputs/GECO-EnglishMaterial.csv'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
df.to_csv(output_path)
print("Saved.")
