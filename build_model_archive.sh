#!/usr/bin/env bash
set -euo pipefail
mkdir -p model_store/distilgpt2
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "distilgpt2"
AutoTokenizer.from_pretrained(name).save_pretrained("model_store/distilgpt2")
AutoModelForCausalLM.from_pretrained(name).save_pretrained("model_store/distilgpt2")
PY
torch-model-archiver \
  --model-name academic_qa \
  --version 1.0 \
  --handler handler.py \
  --serialized-file model_store/distilgpt2 \
  --extra-files "model_store/distilgpt2/config.json,model_store/distilgpt2/generation_config.json,model_store/distilgpt2/merges.txt,model_store/distilgpt2/special_tokens_map.json,model_store/distilgpt2/tokenizer.json,model_store/distilgpt2/tokenizer_config.json,model_store/distilgpt2/vocab.json" \
  --export-path model_store \
  --force
