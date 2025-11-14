import pandas as pd
import re
import string

df = pd.read_csv('Provo_Corpus-Eyetracking_Data.csv', encoding="gbk",
                na_values=["NA", "na", "Na", "N/A", ""],   
    keep_default_na=True                      
)
df = df.dropna(subset=["Text_ID", "Word_Number"])

df = df[['Text_ID','Word_Number','Word']].drop_duplicates().sort_values(['Text_ID','Word_Number'])

df['cut_col'] = df['Text_ID'].astype(str)
df['word'] = df['Word']
df['word_id'] = df['Text_ID'].astype(str) + '-' + df['Word_Number'].astype(str)

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
output_path = 'outputs/PROVO-EnglishMaterial.csv'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
df.to_csv(output_path)
print('Saved.')
