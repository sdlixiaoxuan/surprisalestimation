import pandas as pd
from pathlib import Path
import subprocess, sys
"""
To make a file that incrementally adds a word across lines. Tag the new word.
Context restricted to 1024 words: the minimum context window of models we adopted (GPT-2)
"""
csv_path   = "outputs/GECOCN-ChineseMaterial.csv"           
word_col   = "word"
cut_col     = "cut_col"
sheet_name = 0
max_context = 1024

df = pd.read_csv(csv_path)

stimuli_txt = f"outputs/stimuli_{Path(csv_path).stem}.txt"
full_sentences = []

current_tokens = []
last_cut = None                        

with open(stimuli_txt, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        word = str(row[word_col])
        cut   = row[cut_col]          

        if cut != last_cut:
            current_tokens = []       
            last_cut = cut              

        starred = f"*{word}*"
        sentence_tokens = current_tokens + [starred]
        sentence_tokens = sentence_tokens[-max_context:]
        line = "".join(sentence_tokens)
        fout.write(line + "\n")

        full_sentences.append({               
            "SENTENCE": line,
            "TARGET": word
        })

        current_tokens.append(word)  


print(f"Generated stimuli: {stimuli_txt}")

pd.DataFrame(full_sentences).to_excel(
    f"outputs/stimuli_{Path(csv_path).stem}.xlsx", index=False
)
