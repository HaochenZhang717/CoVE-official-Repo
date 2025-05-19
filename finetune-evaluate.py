import os
import torch
import math
import torch.distributed
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
import wandb
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from src.utils import get_adapter_model, match_module_name, get_wandb_run_name
from emb_comp_utils import get_peft_embedding
import json
from peft import prepare_model_for_kbit_training


@dataclass
class ModelArguments:

    model_name: str = field(metadata={"help": "Huggingface model name"})
    surfix_file_name: str = field(metadata={"help": "epoch"})
    model_path: str = field(default=None, metadata={"help": "Path to the model."})
    adapter_model_path: str = field(default=None, metadata={"help": "Path to the finetuned model."})
    tokenizer_path: str = field(default=None, metadata={"help": "Path to the tokenizer."})
    lora_dim: int = field(default=16, metadata={"help": "The dimension of the adapter."})
    lora_alpha: int = field(default=1, metadata={"help": "The alpha value of the adapter."})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout rate of the adapter."})
    adapter_type: str = field(default="lora", metadata={"help": "One of lora, adalora, dora"})
    lora_init: str = field(default="true",
                           metadata={"help": "true, eva, gaussian, olora, pissa, pissa_niter_[number of iters], loftq"})
    train_embed_only: str = field(default="true",
                           metadata={"help": "true, eva, gaussian, olora, pissa, pissa_niter_[number of iters], loftq"})
    redistribute: bool = field(default=False, metadata={"help": "Wether to redistribute the adapter weights."})
    target_modules: list[str] = field(default=None, metadata={"help": "The target modules for the adapter."})
    ignore_modules: list[str] = field(default=None, metadata={"help": "The modules to ignore."})
    n_components_for_init: int = field(default=None,
                                       metadata={
                                           "help": "The number of components to initialize the adapter with. Remaining components will be initialized randomly"}
                                       )
    model_max_length: int = field(default=None, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    svd_filepath: str = field(default=None, metadata={"help": "Path to the SVD checkpoint file"})
    num_hashes: int = field(default=4, metadata={"help": "number of hashes"})
    compression_rate: float = field(default=8.0, metadata={"help": "compression rate of embedding layer"})
    item_nums: int = field(default=16, metadata={"help": "The number of items."})


@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Path to the training data."})
    dataset_path: str = field(default=None, metadata={
        "help": "Optional local path to the training data. Can be the same as dataset_name."})
    filter_long_context_samples: bool = field(default=False, metadata={
        "help": "Filter out samples with context length > model_max_length."})


def dcg_at_k(ranks, k, label_ids):
    dcg = 0
    for i in range(k):
        rank_i = ranks[i]
        # If rank_i is in the true relevant items, it gets relevance 1
        relevance = 1 if rank_i in [label_ids] else 0
        dcg += relevance / math.log2(i + 2)  # log2(i+2) for 1-indexing
    return dcg


def ndcg_at_k(ranks, k, label_ids):
    dcg = dcg_at_k(ranks, k, label_ids)

    # Calculate IDCG (Ideal DCG), which is 1 if the relevant item is in the top-k
    idcg = 0
    if any(item in ranks[:k] for item in [label_ids]):
        idcg = 1 / math.log2(2)  # Since only one relevant item, the IDCG is either 0 or 1

    return dcg / idcg if idcg > 0 else 0


def hr_at_k(ranks, relevant_items, k):
    # Check if any relevant item is in the top-k
    for rank in ranks[:k]:
        if rank in relevant_items:
            return 1  # Hit
    return 0  # No hit


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    if model_args.model_path is None:
        model_args.model_path = model_args.model_name
    # get around not being able to use multiple types in the same dataclass field
    if model_args.lora_init.lower() == "true":
        model_args.lora_init = True
    # setting this to false to avoid issues with columns that are needed by the data collator
    training_args.remove_unused_columns = False

    if torch.distributed.get_rank() == 0:
        print(model_args)
        print(data_args)
        print(training_args)

    torch.manual_seed(training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_path)
    model = get_peft_embedding(
        model,
        seed=training_args.seed,
        num_hashes=model_args.num_hashes,
        item_nums=model_args.item_nums,
        compression_rate=int(model_args.compression_rate),
    )
    model = get_adapter_model(model, model_args, svd_dict=None, total_steps=None)
    # model.config.vocab_size = model.config.vocab_size + model_args.item_nums
    model.config.vocab_size = model.config.vocab_size + model_args.item_nums + 256
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    model.config.bos_token_id = 128000
    model.config.eos_token_id = 128001

    # model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk

    # adapter_weights = load_file(f'{model_args.adapter_model_path}/model.pt')
    # adapter_weights = torch.load(f'{model_args.adapter_model_path}/model.pt')
    model.to('cpu')
    adapter_weights = torch.load(f'{model_args.adapter_model_path}/model.pt', map_location="cpu")
    # adapter_weights = adapter_weights.to("cuda")

    # for adapter_key, model_key in zip(adapter_weights.keys(), model.state_dict().keys()):
    #     print(f"Adapter weight shape for {adapter_key}: {adapter_weights[adapter_key].shape}")
    #     print(f"Model weight shape for {model_key}: {model.state_dict()[model_key].shape}")
    #     if model_key != adapter_key or adapter_weights[adapter_key].shape != model.state_dict()[model_key].shape:
    #         breakpoint()
    # breakpoint()
    # print(f"adapter weights: {adapter_weights.keys()}")
    # print(f"model weights: {model.state_dict().keys()}")
    # breakpoint()
    # model.load_state_dict(adapter_weights)

    for model_key in model.state_dict().keys():
        if 'lora' in model_key or 'embed' in model_key or 'lm_head' in model_key:
            if model_key in adapter_weights:
                # Ensure the shapes match (this step can be adjusted based on your specific use case)
                model_weight = model.state_dict()[model_key]
                adapter_weight = adapter_weights[model_key]
                if model_weight.shape == adapter_weight.shape:
                    print(f"Loading {model_key} with shape {model_weight.shape}")
                    model.state_dict()[model_key].copy_(adapter_weight)
                else:
                    raise ValueError(f"shape mismatch: {model_key}, model_weight.shape is {model_weight.shape}, adapter_weight.shape is {adapter_weight.shape}")
                    # print(f"Shape mismatch for {model_key}. Skipping...")
            else:
                raise ValueError(f"key mismatch: {model_key}")
    model.cuda()

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]

    result_dict = {
        "NDCG": [],
        "HR": [],
    }

    f = open(f'{data_args.dataset_path}/test_input_only.json', 'r')
    test_data = json.load(f)
    f.close()
    text = [test_datum["text"] for test_datum in test_data]

    f = open(f'{data_args.dataset_path}/test_output_only.json', 'r')
    test_labels = json.load(f)
    f.close()
    label = [test_label["text"] for test_label in test_labels]

    predict_logits = []
    from tqdm import tqdm

    with torch.no_grad():
        for i, batch_input in tqdm(enumerate(batch(text, 1))):
            input = tokenizer(batch_input, return_tensors="pt", padding=True)
            input_ids = input.input_ids
            attention_mask = input.attention_mask
            outputs = model(input_ids.to(model.device), attention_mask=attention_mask.to(model.device), output_hidden_states=True)
            predict_logits.append(outputs.logits[:, -1].detach().cpu())

        predict_logits = torch.cat(predict_logits, dim=0)
    rank = predict_logits[:, 128256:]
    _, rank = torch.sort(rank, dim=-1, descending=True)  # get the rank list

    topk_list = [5, 10]
    NDCG = []
    HRK = []
    for topk in topk_list:
        ndcg_sum = 0
        hrk_sum = 0
        for i in range(len(test_data)):
            label_ids = tokenizer.encode(label[i], add_special_tokens=False)
            assert len(label_ids) == 1
            label_ids = label_ids[0] - 128256
            ndcg = ndcg_at_k(rank[i, :topk], topk, label_ids)
            ndcg_sum += ndcg
            hrk = hr_at_k(rank[i], [label_ids], topk)
            hrk_sum += hrk
        NDCG.append(round(ndcg_sum / len(test_data), 4))
        HRK.append(round(hrk_sum / len(test_data), 4))
    print(NDCG)
    print(HRK)
    result = {'NDCG': NDCG, 'HRK': HRK}
    with open(f"result-{model_args.surfix_file_name}.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
