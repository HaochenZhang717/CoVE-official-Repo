import os
import torch
import math
import torch.distributed
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
import wandb

from src.utils import get_adapter_model, match_module_name, get_wandb_run_name
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict, Dataset
from transformers import DataCollatorForSeq2Seq
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from emb_comp_utils import get_peft_embedding
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



@dataclass
class ModelArguments:

    model_name: str = field(metadata={"help": "Huggingface model name"})
    model_path: str = field(default=None, metadata={"help": "Path to the model."})
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


def main():

    if torch.distributed.is_initialized():
        dist._set_static_graph()


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

    train_dataset = Dataset.from_json(f"{data_args.dataset_path}/train.json")
    # valid_dataset = Dataset.from_json(f"{data_args.dataset_path}/valid.json")
    # test_dataset = Dataset.from_json(f"{data_args.dataset_path}/test.json")

    # Combine into a DatasetDict
    datasets = DatasetDict({
        "train": train_dataset,
        # "validation": valid_dataset,
        # "test": test_dataset
    })

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=model_args.model_max_length
        )
        # Add labels by copying input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    datasets = datasets.map(tokenize_function, batched=True, batch_size=training_args.per_device_train_batch_size)
    datasets = datasets.map(
        lambda examples: {"input_ids": examples["input_ids"]},
        remove_columns=["text"]  # Remove raw text after tokenization
    )

    total_steps = math.ceil(
        len(datasets["train"]) / training_args.per_device_train_batch_size) * training_args.num_train_epochs


    model = get_peft_embedding(
        model,
        seed=training_args.seed,
        num_hashes=model_args.num_hashes,
        item_nums=model_args.item_nums,
        compression_rate=int(model_args.compression_rate)
    )

    model = get_adapter_model(model, model_args, svd_dict=None, total_steps=total_steps)
    model.config.vocab_size = model.config.vocab_size + model_args.item_nums
    # for param in model.base_model.model.model.embed_tokens.item_embed.embedding.parameters():
    #     param.requires_grad = True
    #
    # for param in model.base_model.model.lm_head.item_lm_head.proj.parameters():
    #     param.requires_grad = True


    total_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()  # Get the number of elements in the parameter tensor
    print("Total trainable parameters: {}".format(total_params))

    model.cuda()


    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)

    # print(model.module.state_dict().keys())
    partial_state_dict = {k: v for k, v in model.module.state_dict().items() if 'lora' in k or 'embed' in k or 'lm_head' in k}
    base_state_dict = {k: v for k, v in model.module.state_dict().items() if 'lora' not in k and 'embed' not in k and 'lm_head' not in k}
    # torch.save(partial_state_dict, f'{output_dir}/model.pt')
    # print(f"model saved in {output_dir}")
    # print(partial_state_dict.keys())
    # print(base_state_dict.keys())
    # breakpoint()

    # run name for wandb
    run_name = get_wandb_run_name(
        model_args.model_name,
        data_args.dataset_name,
        model_args.adapter_type,
        model_args.lora_dim,
        model_args.lora_init,
        model_args.redistribute,
        model_args.svd_filepath,
        model_args.n_components_for_init
    )
    # run_name = "compare-with-tiger"

    # save initial adapter state (needed for pissa and olora)
    if torch.distributed.get_rank() == 0:
        wandb_config = {}
        wandb_config.update({f"model_args.{k}": str(v) for k, v in model_args.__dict__.items()})
        wandb_config.update({f"data_args.{k}": str(v) for k, v in data_args.__dict__.items()})
        wandb_config.update({f"training_args.{k}": str(v) for k, v in training_args.__dict__.items()})
        wandb.init(project="Llama3.2-3B-compare-to-tiger", name=run_name, config=wandb_config)

    setattr(training_args, "run_name", run_name)
    print(datasets["train"][0])
    # training_args.prediction_loss_only = True

    class CustomTrainer(Trainer):
        def save_model(self, output_dir=None, _internal_call=False):
            super().save_model(output_dir, _internal_call)
            # partial_state_dict = {k: v for k, v in model.module.state_dict().items() if 'base_model' not in k}
            partial_state_dict = {k: v for k, v in model.module.state_dict().items() if
                                  'lora' in k or 'embed' in k or 'lm_head' in k}
            torch.save(partial_state_dict, f'{output_dir}/model.pt')
            print(f"model saved in {output_dir}")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()
    trainer.save_state()

    if torch.distributed.get_rank() == 0:
        # trainer.model.save_pretrained(training_args.output_dir)
        trainer.model.module.save_pretrained(training_args.output_dir)
        # torch.save(model.module.state_dict(), f'{training_args.output_dir}/model.pt')
        wandb.finish()


if __name__ == "__main__":
    main()
