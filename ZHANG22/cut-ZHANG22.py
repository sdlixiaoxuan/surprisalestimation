import pandas as pd

df = pd.read_csv('ROIs.csv', encoding="gbk")

df['cut_col'] = df['Experiment'].astype(str) + '-' + df['Sentence_ID'].astype(str)
df['word'] = df['Words']
df['word_id'] = df['Experiment'].astype(str) + '-' + df['Sentence_ID'].astype(str) + '-' + df['Word_Order'].astype(str)

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
output_path = 'outputs/ZHANG22-ChineseMaterial.csv'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
df.to_csv(output_path)
print('Saved.')
