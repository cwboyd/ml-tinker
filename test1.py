
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A tiny model that's ~20MB.
model_name = "prajjwal1/bert-tiny"

# Create a tokenizer, download & save that to disk.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(f"cache1/tokenizer/{model_name}")

# Create a model, download & save that to disk.
# These are the weights.
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
model.save_pretrained(f"cache1/model/{model_name}")

# Load back from disk.
tokenizer2 = AutoTokenizer.from_pretrained(f"cache1/tokenizer/{model_name}")
model2 = AutoModelForCausalLM.from_pretrained(f"cache1/model/{model_name}")
