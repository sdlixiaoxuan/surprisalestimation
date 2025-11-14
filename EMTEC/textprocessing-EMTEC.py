import pandas as pd
from pathlib import Path
import subprocess, sys

csv_path   = "outputs/EMTEC-EnglishMaterial.csv"           # 新：你的大 CSV
word_col   = "word"
cut_col     = "cut_col"
sheet_name = 0
max_context = 1024

# --- 读取 CSV ------------------------------------------------
df = pd.read_csv(csv_path)

stimuli_txt = f"outputs/stimuli_{Path(csv_path).stem}.txt"
full_sentences = []

current_tokens = []
last_cut = None                        # ► 初始化上一个 cut_col

with open(stimuli_txt, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        word = str(row[word_col])
        cut   = row[cut_col]            # ► 当前行的 cut_col 值

        # ► 若 cut_col 发生变化，则重置上下文
        if cut != last_cut:
            current_tokens = []       # 清空上下文
            last_cut = cut              # 更新 last_cut

        starred = f"*{word}*"
        sentence_tokens = current_tokens + [starred]
        sentence_tokens = sentence_tokens[-max_context:]
        line = " ".join(sentence_tokens)
        fout.write(line + "\n")

        full_sentences.append({               # 方便后续追溯
            "SENTENCE": line,
            "TARGET": word
        })

        current_tokens.append(word)   # 将纯词加入上下文


print(f"✅ Generated stimuli: {stimuli_txt}")

# --- 保存对照表 ------------------------------------------------
pd.DataFrame(full_sentences).to_excel(
    f"outputs/stimuli_{Path(csv_path).stem}.xlsx", index=False
)
