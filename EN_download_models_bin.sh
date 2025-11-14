#!/bin/bash

set +e

#########
# IF you in China
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
#########

MODELS=(
  "EleutherAI/gpt-j-6b"
)

PATTERNS=('*.bin' '*.json' '*.txt' 'tokenizer.model' '*.py')

MAX_RETRIES=5
RETRY_DELAY=30
COMMAND_TIMEOUT="45m"
MODEL_SLEEP_DELAY=60 

for model_id in "${MODELS[@]}"; do
  echo "=== Now Downloading Model: $model_id ==="
  model_success=true 

  for pattern in "${PATTERNS[@]}"; do
    echo "  --> Downloading: $pattern"
    
    pattern_success=false
    for ((i=1; i<=MAX_RETRIES; i++)); do
      timeout "$COMMAND_TIMEOUT" hf download "$model_id" \
        --include "$pattern" \
        --local-dir "$model_id"
      
      if [ $? -eq 0 ]; then
        echo "  -->  '$pattern' Downloaded"
        pattern_success=true
        break
      else
        echo "  -->  '$pattern' failed ( $i times)。"
        if [ $i -lt $MAX_RETRIES ]; then
          echo "  --> Retry after $RETRY_DELAY "
          sleep $RETRY_DELAY
        else
          echo "  --> Give up to download '$pattern'。"
          model_success=false 
        fi
      fi
    done
    
    if [ "$pattern_success" = false ]; then
      break 
    fi
  done

  if [ "$model_success" = true ]; then
    echo "=== successful downloading: $model_id ==="
  else
    echo "=== Failed to download: $model_id ==="
  fi

  echo ">>> wait $MODEL_SLEEP_DELAY seconds to avoid blocking..."
  sleep "$MODEL_SLEEP_DELAY"
  echo 

done

echo "All downloaded"
