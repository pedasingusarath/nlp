# Run from the project root in PowerShell after installing requirements.
New-Item -ItemType Directory -Force -Path model_store | Out-Null
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('distilgpt2').save_pretrained('model_store/distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2').save_pretrained('model_store/distilgpt2')"

torch-model-archiver `
  --model-name academic_qa `
  --version 1.0 `
  --handler handler.py `
  --serialized-file model_store/distilgpt2 `
  --extra-files "model_store/distilgpt2/config.json,model_store/distilgpt2/generation_config.json,model_store/distilgpt2/merges.txt,model_store/distilgpt2/special_tokens_map.json,model_store/distilgpt2/tokenizer.json,model_store/distilgpt2/tokenizer_config.json,model_store/distilgpt2/vocab.json" `
  --export-path model_store `
  --force
