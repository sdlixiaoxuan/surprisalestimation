import os
import pandas as pd
import numpy as np

# ----------------
# DO NOT contain the merged_surprisal.csv in the directory if you want to use this
# PUT this script in the ./output
current_directory = '.'
output_filename = 'merged_probabilities.csv'
# ----------------

all_files = os.listdir(current_directory)

csv_files = [f for f in all_files if f.endswith('.csv') and f != output_filename]

if not csv_files:
    print("did not find the csv to combine")
else:
    print(f"found: {csv_files}")
    prob_data_to_merge = []
    
    base_df = None

    if csv_files:
        try:
            base_df = pd.read_csv(os.path.join(current_directory, csv_files[0]), usecols=['line_id', 'sentence', 'target'])
        except Exception as e:
            print(f"error when reading {csv_files[0]} 's basic column: {e}")
            base_df = pd.DataFrame() 
    
    for filename in csv_files:
        filepath = os.path.join(current_directory, filename)
        try:
            df = pd.read_csv(filepath, usecols=['surprisal_bits'])
            
            # surprisal -> probability
            #   np.exp2(-x) = 2**(-x)
            df['surprisal_bits'] = np.exp2(-df['surprisal_bits'])
          
            # 'gpt2.csv' -> 'gpt2'
            model_name = os.path.splitext(filename)[0]
            
            # rename
            df.rename(columns={'surprisal_bits': model_name}, inplace=True)
            
            prob_data_to_merge.append(df)
            print(f" Done: {filename}")
            
        except Exception as e:
            print(f"Error when {filename} : {e}")

    if base_df is not None and prob_data_to_merge:
        final_df = pd.concat([base_df] + prob_data_to_merge, axis=1)

        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"file to: {output_filename}")
    else:
        print("no file to combine")
