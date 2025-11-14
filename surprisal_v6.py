#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage
----
python surprisal_v6.py \
    --stimuli corpus/ZHANG22/outputs/stimuli_ZHANG22-ChineseMaterial.txt \
    --models  model_cn.txt \
    --out_dir corpus/ZHANG22/outputs \
    --batch   16 \
    --dtype   fp16

python surprisal_v6.py \
    --stimuli corpus/GECOCN/outputs/stimuli_GECOCN-ChineseMaterial.txt \
    --models  model_cn.txt \
    --out_dir corpus/GECOCN/outputs \
    --batch   16 \
    --dtype   fp16

python surprisal_v6.py \
    --stimuli corpus/HKC/outputs/stimuli_GECOCN-ChineseMaterial.txt \
    --models  model_cn.txt \
    --out_dir corpus/HKC/outputs \
    --batch   16 \
    --dtype   fp16

python surprisal_v6.py \
    --stimuli corpus/MECO/outputs/stimuli_MECO-ChineseMaterial.txt \
    --models  model_en.txt \
    --out_dir corpus/MECO/outputs \
    --batch   16 \
    --dtype   fp16
"""

import os, argparse, csv, math, re, time, gc
from pathlib import Path
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as hf_logging,
)

hf_logging.set_verbosity_info()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 如不在国内可删除
LN2 = math.log(2)


# --------------------------------------------------------------------------- #
#                              Loading Model                    
# --------------------------------------------------------------------------- #
def load_hf_causal(model_name: str,
                   device: str,
                   dtype: str):
    kw = dict(low_cpu_mem_usage=True, trust_remote_code=True)
    if dtype == "fp16" and device.startswith("cuda"):
        kw.update(torch_dtype=torch.float16)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True) 
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)

    print(f"  - Moving model to {device}...")
    model.to(device)
    print("  - Model moved successfully.")

    if tok.pad_token_id is None:
        print(f"  - Tokenizer for {model_name} lacks a pad_token. Adding '[PAD]' as a new special token.")
        tok.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tok))

    pad_id = tok.pad_token_id

    # The max context is set when we preprocess the materials.
    max_len = (
        getattr(model.config, "n_positions", None) or
        getattr(model.config, "max_position_embeddings", None) or
        getattr(model.config, "n_ctx", None)
        or 2048
    )
    bos_id = tok.bos_token_id or tok.cls_token_id or tok.eos_token_id
    if bos_id is None:
        raise ValueError(f"{model_name} no BOS/CLS/EOS token")

    model.eval()
    return tok, model, max_len, bos_id, pad_id

# --------------------------------------------------------------------------- #
#                 surprisal (causal-LM) calculation                  
# --------------------------------------------------------------------------- #
@torch.no_grad()
def surp_causal(model, tok,
                bos_id: int, pad_id: int,
                sents_with_star: List[str],
                device: str, max_len: int,
                use_amp: bool):
    
    pat = re.compile(r"\*(.*?)\*")
    
    all_input_ids = []
    target_token_indices = [] 

    for sent in sents_with_star:
        m = pat.search(sent)
        if not m:
            print(f"Warning: No target word with '*' found in '{sent}'. Skipping.")
            all_input_ids.append(None)
            continue

        target_text = m.group(1)
        clean_sent = sent.replace("*", "")
        char_start = m.start()
        char_end = char_start + len(target_text)

        encoding = tok(clean_sent, return_offsets_mapping=True, add_special_tokens=False)
        full_ids = [bos_id] + encoding['input_ids']
        offsets = [(0, 0)] + encoding['offset_mapping']

        start_token_idx = -1
        end_token_idx = -1

        for i, (offset_start, offset_end) in enumerate(offsets):
            # The first token whose end offset is after the target's start char is the start token
            if offset_end > char_start:
                start_token_idx = i
                break
        
        for i in range(len(offsets) - 1, -1, -1):
            offset_start, offset_end = offsets[i]
            # The first token (from the end) whose start offset is before the target's end char is the end token
            if offset_start < char_end:
                # The end index for slicing should be exclusive, so +1
                end_token_idx = i + 1
                break

        if start_token_idx == -1 or end_token_idx == -1 or start_token_idx >= end_token_idx:
            print(f"Warning: Could not map target '{target_text}' in '{clean_sent}' to tokens. Skipping.")
            all_input_ids.append(None)
            continue
        
        # cutoff: already done in the preprocessing
        if len(full_ids) > max_len:
            cutoff = len(full_ids) - max_len
            if start_token_idx < cutoff:
                print(f"Warning: Target word for '{clean_sent}' is truncated due to max_len. Skipping.")
                all_input_ids.append(None)
                continue
            
            full_ids = full_ids[cutoff:]
            start_token_idx -= cutoff
            end_token_idx -= cutoff

        all_input_ids.append(full_ids)
        target_token_indices.append((start_token_idx, end_token_idx))

    # ----- 2. calculation -----
    valid_indices = [i for i, ids in enumerate(all_input_ids) if ids is not None]
    if not valid_indices:
        return [float('nan')] * len(sents_with_star)

    valid_input_ids = [all_input_ids[i] for i in valid_indices]
    
    tensors = [torch.tensor(s, dtype=torch.long) for s in valid_input_ids]
    input_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_id).to(device)
    attn = input_ids.ne(pad_id)

    with torch.amp.autocast(device_type=device.split(':')[0], enabled=use_amp):
        logits = model(input_ids, attention_mask=attn, use_cache=False).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # ----- 3. Surprisal calculation -----
    batch_surps = []
    for i, valid_idx in enumerate(valid_indices):
        start_idx, end_idx = target_token_indices[valid_idx]
        
        target_tokens = valid_input_ids[i][start_idx:end_idx]

        total_log_prob = 0.0
        for j, token_id in enumerate(target_tokens):
            prediction_pos = start_idx + j - 1
            total_log_prob += log_probs[i, prediction_pos, token_id].item()
        
        batch_surps.append(-total_log_prob / LN2)
    
    final_surps = []
    surp_idx = 0
    for i in range(len(sents_with_star)):
        if all_input_ids[i] is None:
            final_surps.append(float('nan'))
        else:
            final_surps.append(batch_surps[surp_idx])
            surp_idx += 1
            
    return final_surps



# --------------------------------------------------------------------------- #
#                               Main Processing                               #
# --------------------------------------------------------------------------- #
def run_one_model(model_name: str,
                  lines_with_star: List[str], 
                  out_dir: Path,
                  batch: int,
                  device: str,
                  dtype: str):

    tok, mdl, max_len, bos_id, pad_id = load_hf_causal(model_name, device, dtype)
    clean_name = model_name.replace("/", "-")
    csv_path = out_dir / f"{clean_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    pat = re.compile(r"\*(.*?)\*")
    clean_lines = [s.replace("*","") for s in lines_with_star]
    target_words = [pat.search(s).group(1) for s in lines_with_star]

    all_surps = []
    with csv_path.open("w", newline="", encoding="utf-8") as fw:
        wr = csv.writer(fw)
        wr.writerow(["line_id", "sentence", "target", "surprisal_bits", "n_subtokens"])

        for i in range(0, len(lines_with_star), batch):
            batch_sents = lines_with_star[i : i+batch]
            
            surps = surp_causal(
                mdl, tok, bos_id, pad_id,
                batch_sents,
                device, max_len,
                use_amp=(dtype == "fp16" and device.startswith("cuda"))
            )
            all_surps.extend(surps)
            
            for j, s_val in enumerate(surps):
                original_idx = i + j
                target_word = target_words[original_idx]
                n_subtokens = len(tok.encode(target_word, add_special_tokens=False))
                wr.writerow([
                    original_idx + 1,
                    clean_lines[original_idx],
                    target_word,
                    s_val,
                    n_subtokens
                ])

    print(f"  ✔ {model_name} → {csv_path}")
    del mdl, tok
    gc.collect(); torch.cuda.empty_cache()
    return all_surps


def main():
    ap = argparse.ArgumentParser("calculating surprisal(bits) (causal LMs)")
    ap.add_argument("--stimuli", "-i", required=True,
                    help="Each line contains one sentence. tag by *word*")
    ap.add_argument("--models", "-m", required=True,
                    help="txt file for model list. each line for one model")
    ap.add_argument("--out_dir", "-o", default="surprisal_outputs",
                    help="output directory")
    ap.add_argument("--batch", type=int, default=32, help="batch size")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--cpu", action="store_true", help="using CPU")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- stimuli compile----
    with open(args.stimuli, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f] 

    targets_for_output = []
    sentences_for_output = []
    pat = re.compile(r"\*(.*?)\*")
    for ln in lines:
        m = pat.search(ln)
        if not m:
            raise ValueError(f"lack *word*：{ln}")
        targets_for_output.append(m.group(1))
        sentences_for_output.append(ln.replace("*", ""))

    # ---- read model list ----
    out_dir = Path(args.out_dir)
    with open(args.models, encoding="utf-8") as f:
        models = [l.strip() for l in f if l.strip()]
    print(f"[Info] Models numbers: {len(models)} ")

    all_surps, meta_df = {}, None
    for idx, mdl in enumerate(models, 1):
        print(f"\n=== [{idx}/{len(models)}] loading {mdl} ===")
        t0 = time.time()
        try:
            surps = run_one_model(
                mdl, lines,
                out_dir, args.batch, device, args.dtype
            )
        except Exception as e:
            print(f"[Error] {mdl} fail：{e}")
            continue
        all_surps[mdl] = surps
        print(f"    {mdl} done in {time.time()-t0:.1f}s")

        if meta_df is None:           
            import pandas as pd
            pat = re.compile(r"\*(.*?)\*")
            targets = [pat.search(l).group(1) for l in lines]
            
            meta_df = pd.DataFrame({
                "line_id": range(1, len(lines)+1),
                "sentence": [l.replace("*", "") for l in lines],
                "target": targets
            })

    if meta_df is None:
        print("all wrong, exit")
        return

    for mdl, col in all_surps.items():
        meta_df[mdl.replace("/", "_")] = col

    merged = out_dir / "merged_surprisal.csv"
    merged.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(merged, index=False, encoding="utf-8")
    print(f"\n✅ all results to: {merged}")


if __name__ == "__main__":
    main()
