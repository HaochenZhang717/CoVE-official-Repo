import json
from transformers import AutoTokenizer
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Llama-2-7b-hf", help="Model ID or path")
parser.add_argument("--data_name", type=str, default="beauty", help="Dataset name")
args = parser.parse_args()

# tokenizer_path = f"./{args.model_id}/orig"
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
product_file = f"../dataset/amazon/raw/{args.data_name}/datamaps.json"

with open(product_file, "r") as file:
    metadata = json.load(file)
    metadata = metadata["item2id"]

new_special_tokens = []
for asin in metadata.values():
    new_special_tokens.append(f"<|{asin}|>")

# Add new special tokens
tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
print("Special Tokens Map:", tokenizer.special_tokens_map)
print("New Tokens:", tokenizer.additional_special_tokens)

tokenizer.save_pretrained(f'{args.model_id}/{args.data_name}/tokenizer')

text = "<|11|>"
tokens = tokenizer(text, return_tensors="pt")
print(tokens)
