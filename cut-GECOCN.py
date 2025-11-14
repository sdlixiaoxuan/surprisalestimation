import pandas as pd
"""
To ensure the context is indeed in one sentence/passage, we need to clip when the material changes.
"""
df = pd.read_excel('ChineseMaterial.xlsx')

df['cut_col'] = df['WORD_ID'].str[0]
df['word_id'] = df['WORD_ID']

dfword = pd.read_excel('L1ReadingData.xlsx')
col_to_keep = [
    'WORD_ID',
    'IA_LABEL'
]
dfword1 = dfword[col_to_keep].drop_duplicates(subset=['WORD_ID'])
dfword1['word'] = dfword1['IA_LABEL']

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
output_path = 'outputs/GECOCN-ChineseMaterial.csv'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
df.to_csv(output_path)
print("Saved.")