# -*- coding: utf-8 -*-
"""
数据预处理脚本

功能:
1. 读取并合并文本信息和surprisal数据。
2. 读取并合并词频数据。
3. 计算滞后变量 (前一个词和前两个词的词频、词长、surprisal)。
4. 在指定的边界条件下 (如句子ID或被试ID改变)，重置滞后变量的计算。
5. 保存最终处理好的DataFrame。
"""
import pandas as pd
import numpy as np

# ===============================================================
# 1. 文件路径设置 (在这里修改您的文件路径)
# ===============================================================

# 输入文件路径
# 假设surprisal文件是一个简单的文本文件，每行一个surprisal值
SURPRISAL_FILE_PATH = 'outputs/merged_surprisal_with_pca.csv' 

# 假设文本信息文件是一个CSV文件
TEXT_INFO_FILE_PATH = 'outputs/PROVO-EnglishMaterial.csv'

# 假设词频文件是一个CSV文件
WORD_FREQ_FILE_PATH = 'SUBTLEX-US.xlsx'

# 输出文件路径
PROCESSED_DATA_OUTPUT_PATH = 'outputs/processed_experiment_data.csv'

# Reading Data Path
READING_DATA_PATH = 'Provo_Corpus-Eyetracking_Data.csv'
READING_DATA_OUTPUT_PATH = 'outputs/DATA_PROVO.csv'


# ===============================================================
# 2. 列名设置 (在这里根据您的文件修改列名)
# ===============================================================

# 文本信息文件中的列名
# 假设文本信息文件包含一个'word'列用于合并词频
# 和一个'boundary_col'用于判断边界
TEXT_INFO_WORD_COL = 'word' #Generated from cut-.py. For Frequency measure
BOUNDARY_COL = 'cut_col' #Generated from cut-.py. For lag measure
WORDID_COL = 'word_id' #Generated from cut-.py. For combining with Eye movement data

# 词频文件中的列名
# 假设词频文件包含一个'word'列和一个'frequency'列
FREQ_FILE_WORD_COL = 'Word'
FREQ_FILE_FREQ_COL = 'Lg10WF'

# 眼动数据文件中的列名
EYE_DATA_CONDITION_COL = 'IA_FIRST_FIX_PROGRESSIVE'
EYE_DATA_SOURCE_FFD = 'IA_FIRST_FIXATION_DURATION'
# ===============================================================
# 3. 数据加载与初步合并(获取surprisal，计算词频)
# ===============================================================

print("--- 开始数据预处理 ---")

try:
    # 读取 surprisal 文件 (假设为单列无表头)
    print(f"正在读取 surprisal 文件: {SURPRISAL_FILE_PATH}")
    surprisal_df = pd.read_csv(SURPRISAL_FILE_PATH)
    surprisal_df['averaged_prob'] = surprisal_df['PC1_prob']
    print(f"成功读取 {len(surprisal_df)} 行 surprisal 数据。")

    # 读取文本信息文件
    print(f"正在读取文本信息文件: {TEXT_INFO_FILE_PATH}")
    text_info_df = pd.read_csv(TEXT_INFO_FILE_PATH)
    print(f"成功读取 {len(text_info_df)} 行文本信息数据。")

    # 检查行数是否匹配
    if len(surprisal_df) != len(text_info_df):
        raise ValueError(
            f"Surprisal文件行数 ({len(surprisal_df)}) 与文本信息文件行数 ({len(text_info_df)}) 不匹配!"
        )

    # 直接横向合并 (顺序是对齐的，基于textprocessing和surprisal_v1逻辑)
    print("正在合并 surprisal 和文本信息...")
    df_main = pd.concat([text_info_df, surprisal_df], axis=1)
    print("合并完成。")

    # 读取词频文件 (只取需要的两列)
    print(f"正在读取词频文件: {WORD_FREQ_FILE_PATH}")
    freq_df = pd.read_excel(
        WORD_FREQ_FILE_PATH, 
        usecols=[FREQ_FILE_WORD_COL, FREQ_FILE_FREQ_COL]
    )
    # 为了避免合并时的列名冲突，可以重命名
    freq_df.rename(columns={
        FREQ_FILE_WORD_COL: TEXT_INFO_WORD_COL,
        FREQ_FILE_FREQ_COL: 'freq'
        }, inplace=True)
    print(f"成功读取 {len(freq_df)} 行词频数据。")

    # 将词频合并到主DataFrame
    print("正在根据词语合并词频数据...")
    df_full = pd.merge(df_main, freq_df, on=TEXT_INFO_WORD_COL, how='left')
    print("词频合并完成。")
    
    # 检查合并后是否有缺失的词频
    missing_freq_count = df_full['freq'].isnull().sum()
    if missing_freq_count > 0:
        print(f"警告: 有 {missing_freq_count} 个词语未能匹配到词频，其词频将为NaN。")

except FileNotFoundError as e:
    print(f"错误: 文件未找到 - {e}")
    exit()
except Exception as e:
    print(f"发生错误: {e}")
    exit()

# ===============================================================
# 4. 计算词长
# ===============================================================
print("正在计算词长...")
# 使用.str.len()计算字符串长度，对非字符串值(如NaN)返回NaN
df_full['len'] = df_full[TEXT_INFO_WORD_COL].str.len()
print("词长计算完成。")


# ===============================================================
# 5. 计算滞后变量
# ===============================================================
print(f"正在根据边界列 '{BOUNDARY_COL}' 计算滞后变量...")

# 定义需要计算滞后项的列
cols_to_lag = ['freq', 'len', 'averaged_prob']

# 使用 groupby() 和 shift() 的组合拳来安全地计算滞后
# groupby(BOUNDARY_COL) 确保了计算不会跨越边界
# 例如，每个句子的第一行的 shift(1) 结果会自动成为 NaT/NaN
for col in cols_to_lag:
    if BOUNDARY_COL == 'none':
        df_full[f'{col}_lag1'] = df_full[col].shift(1)
        df_full[f'{col}_lag2'] = df_full[col].shift(2)
    else:
        # 计算前一个词 (lag 1)
        df_full[f'{col}_lag1'] = df_full.groupby(BOUNDARY_COL)[col].shift(1)
        # 计算前两个词 (lag 2)
        df_full[f'{col}_lag2'] = df_full.groupby(BOUNDARY_COL)[col].shift(2)

print("滞后变量计算完成。")


# ===============================================================
# 6. 保存结果
# ===============================================================
print(f"正在将最终结果保存到: {PROCESSED_DATA_OUTPUT_PATH}")
try:
    # 确保输出目录存在
    import os
    output_dir = os.path.dirname(PROCESSED_DATA_OUTPUT_PATH)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
        
    df_full.to_csv(PROCESSED_DATA_OUTPUT_PATH, index=False, encoding='utf-8-sig')


except Exception as e:
    print(f"保存文件时发生错误: {e}")


# ===============================================================
# 7. Combine with Reading Data
# ===============================================================

print("\n--- 开始处理眼动数据 ---")
try:
    print(f"正在读取眼动数据: {READING_DATA_PATH}")
    ReadingData = pd.read_csv(READING_DATA_PATH)

    # 为了与材料文件合并，确保key的列名一致
    #Same as cut-.py
    ReadingData[WORDID_COL] = ReadingData['Text_ID'].astype(str) + '-' + ReadingData['Word_Number'].astype(str) #Same as word_id in cut-.py
    print(f"成功读取 {len(ReadingData)} 行眼动数据。")

    print("正在根据 'word_id' 合并材料与眼动数据...")
    ReadingData_full = pd.merge(ReadingData, df_full, on=WORDID_COL, how='left')
    print("合并完成。")
    
    missing_material_count = ReadingData_full[TEXT_INFO_WORD_COL].isnull().sum()
    if missing_material_count > 0:
        print(f"警告: 有 {missing_material_count} 行眼动数据未能匹配到材料信息。")

except FileNotFoundError as e:
    print(f"错误: 文件未找到 - {e}")
    exit()
except Exception as e:
    print(f"合并眼动数据时发生错误：{e}")
    exit()

# ===============================================================
# 8. 计算关键眼动指标 (FFD, GD)
# ===============================================================
print("正在计算 FFD (First Fixation Duration) 和 GD (Gaze Duration)...")

try:
    
    
    ReadingData_full[EYE_DATA_SOURCE_FFD] = pd.to_numeric(ReadingData_full[EYE_DATA_SOURCE_FFD], errors='coerce')
    ReadingData_full['IA_FIRST_RUN_DWELL_TIME'] = pd.to_numeric(ReadingData_full['IA_FIRST_RUN_DWELL_TIME'], errors='coerce')
    # 检查转换后是否有值变成了NaN，这表示原始数据有问题
    coerced_nan_count = ReadingData_full[['IA_FIRST_RUN_DWELL_TIME']].isnull().sum().sum()
    if coerced_nan_count > 0:
        print(f"警告: 在转换时间列为数值时，有 {coerced_nan_count} 个值因格式问题被强制转换为NaN。")
    
    # 确保条件列是字符串类型，以便进行比较
    # 使用 .astype(str) 可以安全地处理可能存在的非字符串值
    condition_series = ReadingData_full[EYE_DATA_CONDITION_COL].astype(str)

    # 方法1: 使用 np.where (推荐，非常高效)
    # np.where(condition, value_if_true, value_if_false)
    # 计算 FFD
    ReadingData_full['FFD'] = np.where(
        ReadingData_full['IA_FIRST_FIX_PROGRESSIVE'].astype(str) == '1.0',                      # 条件
        ReadingData_full['IA_FIRST_FIXATION_DURATION'],       # 如果为真，取此列的值
        np.nan                                       # 如果为假，设为NaN
    )

    # 计算 GD
    ReadingData_full['GD'] = np.where(
        ReadingData_full['IA_FIRST_FIX_PROGRESSIVE'].astype(str) == '1.0',                      # 条件
        ReadingData_full['IA_FIRST_RUN_DWELL_TIME'] , # 如果为真，取此值
        np.nan                                       # 如果为假，设为NaN
    )
    ReadingData_full['SFD'] = np.where(
        ReadingData_full['FFD'] == ReadingData_full['GD'],                      # 条件
        ReadingData_full['FFD'], # 如果为真，取此值
        np.nan                                       # 如果为假，设为NaN
    )
    ReadingData_full['TT'] = ReadingData_full['IA_DWELL_TIME']
    ReadingData_full['Gopast'] = ReadingData_full['IA_REGRESSION_PATH_DURATION']

    
    
    print("FFD, GD, SFD, TT, Gopast 计算完成。")
    
    # 报告一下计算结果
    valid_ffd_count = ReadingData_full['FFD'].notna().sum()
    valid_gd_count = ReadingData_full['GD'].notna().sum()
    print(f"共计算出 {valid_ffd_count} 个有效的 FFD 值和 {valid_gd_count} 个有效的 GD 值。")

except KeyError as e:
    print(f"错误: 在DataFrame中找不到指定的列名 - {e}")
    print("请检查第2部分的列名设置是否正确。")
    exit()
except Exception as e:
    print(f"计算眼动指标时发生错误: {e}")
    exit()

# ===============================================================
# 9. 筛选最终列并保存结果
# ===============================================================

# --- 在这里定义您最终想要保留的所有列名 ---
# 这是一个示例列表，请根据您的分析需求进行修改
# 我包含了一些常见的标识符、因变量、自变量和协变量

ReadingData_full.rename(columns={'FFD':'FFD',
                                 'GD':'GD',
                                 'averaged_prob':'prob',
                                 'averaged_prob_lag1':'prob_lag1',
                                 'averaged_prob_lag2':'prob_lag2',
                                 'word_id': 'word_id',
                                 'Participant_ID':'sub'
                                }, inplace=True)
final_columns_to_keep = [
    # 标识符
    'sub',
    'word_id',
    'word',
    
    # 因变量 (眼动指标)
    'FFD',
    'GD',
    'SFD',
    'TT',
    'Gopast',
    
    # 主要自变量
    'prob',
    
    # 协变量
    'freq',
    'len',
    
    # 滞后变量
    'freq_lag1', 'len_lag1', 'prob_lag1',
    'freq_lag2', 'len_lag2', 'prob_lag2'
]

# 检查所有想要的列是否存在于DataFrame中
missing_cols = [col for col in final_columns_to_keep if col not in ReadingData_full.columns]
if missing_cols:
    print(f"\n警告: 以下您想保留的列在最终的DataFrame中不存在: {missing_cols}")
    print("它们将被忽略。请检查列名是否拼写正确。")
    # 从列表中移除不存在的列，以避免错误
    final_columns_to_keep = [col for col in final_columns_to_keep if col in ReadingData_full.columns]

print(f"\n将筛选数据，只保留 {len(final_columns_to_keep)} 个指定的列。")
final_df = ReadingData_full[final_columns_to_keep]

print(f"正在将最终的筛选后数据(不去除了任何NA)保存到: {READING_DATA_OUTPUT_PATH}，一共{final_df.shape[0]}行")
try:
    final_df.to_csv(READING_DATA_OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print("\n--- 全部预处理成功完成！ ---")
    print("\n最终数据概览 (前5行):")
    print(final_df.head())
    print("\n最终数据列名:")
    print(final_df.columns.tolist())
except Exception as e:
    print(f"保存最终文件时发生错误: {e}")
    

