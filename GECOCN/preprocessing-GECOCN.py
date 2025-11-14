# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np

# ===============================================================
# 1. Paths
# ===============================================================

SURPRISAL_FILE_PATH = 'outputs/merge/merged_probabilities_with_avg.csv' 

TEXT_INFO_FILE_PATH = 'outputs/GECOCN-ChineseMaterial.csv'

WORD_FREQ_FILE_PATH = 'SUBTLEX-CH-WF.xlsx'

PROCESSED_DATA_OUTPUT_PATH = 'outputs/processed_experiment_data.csv'

READING_DATA_PATH = 'L1ReadingData.xlsx'
READING_DATA_OUTPUT_PATH = 'outputs/merge/DATA_GECOCN.csv'


# ===============================================================
# 2. Colnames
# ===============================================================

TEXT_INFO_WORD_COL = 'word' #Generated from cut-.py. For Frequency measure
BOUNDARY_COL = 'cut_col' #Generated from cut-.py. For lag measure
WORDID_COL = 'word_id' #Generated from cut-.py. For combining with Eye movement data

FREQ_FILE_WORD_COL = 'Word'
FREQ_FILE_FREQ_COL = 'logW'

EYE_DATA_CONDITION_COL = 'IA_FIRST_FIX_PROGRESSIVE'
EYE_DATA_SOURCE_FFD = 'IA_FIRST_FIXATION_DURATION'

# ===============================================================
# 3. Data loading
# ===============================================================

print("--- Start Preprocessing ---")

try:
    print(f"Now reading Surprisal data: {SURPRISAL_FILE_PATH}")
    surprisal_df = pd.read_csv(SURPRISAL_FILE_PATH)
    surprisal_df['averaged_prob'] = surprisal_df['avg_probability']

    print(f"Now reading material data: {TEXT_INFO_FILE_PATH}")
    text_info_df = pd.read_csv(TEXT_INFO_FILE_PATH)

    if len(surprisal_df) != len(text_info_df):
        raise ValueError(
            f"Lines of surprisal data: ({len(surprisal_df)}) and material data: ({len(text_info_df)}) did not equal"
        )

    df_main = pd.concat([text_info_df, surprisal_df], axis=1)
    df_main[TEXT_INFO_WORD_COL] = df_main[TEXT_INFO_WORD_COL].str.strip().str.replace(r"[^\w\s'-]", "", regex=True)

    print(f"Now reading frequency data: {WORD_FREQ_FILE_PATH}")
    freq_df = pd.read_excel(
        WORD_FREQ_FILE_PATH, 
        usecols=[FREQ_FILE_WORD_COL, FREQ_FILE_FREQ_COL]
    )
    freq_df.rename(columns={
        FREQ_FILE_WORD_COL: TEXT_INFO_WORD_COL,
        FREQ_FILE_FREQ_COL: 'freq'
        }, inplace=True)

    df_full = pd.merge(df_main, freq_df, on=TEXT_INFO_WORD_COL, how='left')
    
    missing_freq_count = df_full['freq'].isnull().sum()
    if missing_freq_count > 0:
        print(f" {missing_freq_count} of word in material did not show in Frequency data, NaN")

except Exception as e:
    print(f"error: {e}")
    exit()

# ===============================================================
# 4. Word Length
# ===============================================================

df_full['len'] = df_full[TEXT_INFO_WORD_COL].str.len()

# ===============================================================
# 5. Lag Variables
# ===============================================================
print(f"calculating lag variables according to: '{BOUNDARY_COL}' ")

cols_to_lag = ['freq', 'len', 'averaged_prob']

for col in cols_to_lag:
    if BOUNDARY_COL == 'none':
        df_full[f'{col}_lag1'] = df_full[col].shift(1)
        df_full[f'{col}_lag2'] = df_full[col].shift(2)
    else:
        # 计算前一个词 (lag 1)
        df_full[f'{col}_lag1'] = df_full.groupby(BOUNDARY_COL)[col].shift(1)
        # 计算前两个词 (lag 2)
        df_full[f'{col}_lag2'] = df_full.groupby(BOUNDARY_COL)[col].shift(2)

# ===============================================================
# 6. Save the data without Eye-movement data
# ===============================================================

print(f"Save the data without Eye-movement measures to: {PROCESSED_DATA_OUTPUT_PATH}")
try:
    # 确保输出目录存在
    import os
    output_dir = os.path.dirname(PROCESSED_DATA_OUTPUT_PATH)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
        
    df_full.to_csv(PROCESSED_DATA_OUTPUT_PATH, index=False, encoding='utf-8-sig')

except Exception as e:
    print(f"error: {e}")

# ===============================================================
# 7. Combine with Reading Data
# ===============================================================

try:
    print(f"Now reading eye-movement data: {READING_DATA_PATH}")
    ReadingData = pd.read_excel(READING_DATA_PATH)

    #Same as cut-.py
    ReadingData[WORDID_COL] = ReadingData['WORD_ID']

    ReadingData_full = pd.merge(ReadingData, df_full, on='word_id', how='left')
    
    missing_material_count = ReadingData_full[TEXT_INFO_WORD_COL].isnull().sum()
    if missing_material_count > 0:
        print(f" {missing_material_count}  lines of eye-movement data did not find the material information")

except FileNotFoundError as e:
    print(f"error: file path wrong - {e}")
    exit()
except Exception as e:
    print(f"error: {e}")
    exit()

# ===============================================================
# 8. Eye-movement Measures calculation
# ===============================================================
try:
    ReadingData_full[EYE_DATA_SOURCE_FFD] = pd.to_numeric(ReadingData_full[EYE_DATA_SOURCE_FFD], errors='coerce')
    ReadingData_full['IA_FIRST_RUN_START_TIME'] = pd.to_numeric(ReadingData_full['IA_FIRST_RUN_START_TIME'], errors='coerce')
    ReadingData_full['IA_FIRST_RUN_END_TIME'] = pd.to_numeric(ReadingData_full['IA_FIRST_RUN_END_TIME'], errors='coerce')
    
    coerced_nan_count = ReadingData_full[['IA_FIRST_RUN_START_TIME', 'IA_FIRST_RUN_END_TIME']].isnull().sum().sum()
    if coerced_nan_count > 0:
        print(f" {coerced_nan_count}  values to NaN because the format")
    
    condition_series = ReadingData_full[EYE_DATA_CONDITION_COL].astype(str)

    ReadingData_full['FFD'] = np.where(
        condition_series == '1',                      
        ReadingData_full[EYE_DATA_SOURCE_FFD],       
        np.nan                                       
    )

    ReadingData_full['GD'] = np.where(
        condition_series == '1',                      
        ReadingData_full['IA_FIRST_RUN_END_TIME'] - ReadingData_full['IA_FIRST_RUN_START_TIME'], 
        np.nan                                       
    )
    
    ReadingData_full['SFD'] = np.where(
        abs(ReadingData_full['FFD'] - ReadingData_full['GD']) < 2,                     
        ReadingData_full['FFD'], 
        np.nan                                      
    )
    ReadingData_full['TT'] = ReadingData_full['IA_DWELL_TIME']
    ReadingData_full['Gopast'] = ReadingData_full['IA_REGRESSION_PATH_DURATION']

    print("FFD, GD, SFD, TT, Gopast 计算完成。")
    
    valid_ffd_count = ReadingData_full['FFD'].notna().sum()
    valid_gd_count = ReadingData_full['GD'].notna().sum()

except KeyError as e:
    print(f"error: did not found the colnames in eye-movement data {e}")
    print("Check the setting in part 2")
    exit()
except Exception as e:
    print(f"error: {e}")
    exit()

# ===============================================================
# 9. Keep the cols we need, Rename
# ===============================================================
ReadingData_full.rename(columns={'FFD':'FFD',
                                 'GD':'GD',
                                 'averaged_prob':'prob',
                                 'averaged_prob_lag1':'prob_lag1',
                                 'averaged_prob_lag2':'prob_lag2',
                                 'word_id': 'word_id',
                                 'PP_ID':'sub'
                                }, inplace=True)
final_columns_to_keep = [
    # sub and item
    'sub',
    'word_id',
    'word',
    
    # measures
    'FFD',
    'GD',
    'SFD',
    'TT',
    'Gopast',
    
    # prob
    'prob',
    
    # covariates
    'freq',
    'len',
    
    # lag_covariates
    'freq_lag1', 'len_lag1', 'prob_lag1',
    'freq_lag2', 'len_lag2', 'prob_lag2'
]

missing_cols = [col for col in final_columns_to_keep if col not in ReadingData_full.columns]
if missing_cols:
    print(f"\ndid not find this cols: {missing_cols}")
    print("Check if the spell is right")
    final_columns_to_keep = [col for col in final_columns_to_keep if col in ReadingData_full.columns]

final_df = ReadingData_full[final_columns_to_keep]

print(f"Data save to: {READING_DATA_OUTPUT_PATH}, {final_df.shape[0]} lines")
try:
    final_df.to_csv(READING_DATA_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(final_df.head())
    print(final_df.columns.tolist())
except Exception as e:
    print(f"error: {e}")
    

