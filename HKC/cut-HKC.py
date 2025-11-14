import pandas as pd

df = pd.read_csv('data_all.csv')

df = df[['Format','SENTENCE_OR_PASSAGE_NUMBER','IP_INDEX','IA_ID','IA_LABEL']].drop_duplicates().sort_values(['Format','SENTENCE_OR_PASSAGE_NUMBER','IP_INDEX','IA_ID'])

df['cut_col'] = df['Format'].astype(str) + '-' + df['SENTENCE_OR_PASSAGE_NUMBER'].astype(str) 
df['word'] = df['IA_LABEL']
df['word_id'] = df['Format'].astype(str) + '-' + df['SENTENCE_OR_PASSAGE_NUMBER'].astype(str) + '-' + df['IP_INDEX'].astype(str) + '-' + df['IA_ID'].astype(str)

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
output_path = 'outputs/HKC-ChineseMaterial.csv'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
df.to_csv(output_path)
print('Saved.')
