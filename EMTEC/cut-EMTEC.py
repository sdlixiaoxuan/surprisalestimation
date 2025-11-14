import pandas as pd
import re
import string

df = pd.read_csv('reading_measures_corrected.csv', encoding="gbk",
                na_values=["NA", "na", "Na", "N/A", ""],   
    keep_default_na=True, sep='\t')

df = df[['item_id','TRIAL_ID','model','decoding_strategy','word_id','word']].drop_duplicates().sort_values(['item_id','TRIAL_ID','model','decoding_strategy','word_id'])

df['cut_col'] = df['item_id'].astype(str) + '-' +  df['model'].astype(str) + '-' + df['decoding_strategy'].astype(str) +  '-' + df['TRIAL_ID'].astype(str)
df['word'] = df['word']
df['word_id'] = df['item_id'].astype(str) + '-' + df['model'].astype(str) + '-' + df['decoding_strategy'].astype(str) + '-' + df['TRIAL_ID'].astype(str) + '-' + df['word_id'].astype(str)

col_to_keep = [
    'cut_col',
    'word',
    'word_id'
]
df = df[col_to_keep]

print(df.head())
print(f"Words number: {df.shape[0]}")
print(f"Total cut into: {df['cut_col'].unique()}")

import os
output_path = 'outputs/EMTEC-EnglishMaterial.csv'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
df.to_csv(output_path)
print('Saved.')
