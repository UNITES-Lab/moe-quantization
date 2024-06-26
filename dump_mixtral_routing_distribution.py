# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/15
import os
import random

import torch
from datasets import load_dataset, Dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    set_seed,
    MixtralForCausalLM,
    AutoTokenizer,
    default_data_collator
)

set_seed(42)


def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(f"Invalid split: {split}")
    # length of 288059 should be enough
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def dump_mixtral_routing_top_trace(
        save_dir: str = "./results",
        batch_size: int = 1
):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    dataset = get_wikitext2(tokenizer=tokenizer, seqlen=4096, nsamples=512, split="train")
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16, device_map='auto'
    )
    config = model.config

    expert_routed_distribution = torch.zeros(config.num_hidden_layers, config.num_local_experts)

    for batch in tqdm(data_loader, desc=f"Dumping routing distribution"):
        batch = {k: v.cuda() for k, v in batch.items()}
        if "labels" in batch:
            batch.pop("labels")
        if batch_size == 1:
            for k, v in batch.items():
                batch[k] = v.squeeze(0)
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True)
        all_router_logits = outputs.router_logits
        all_router_logits = torch.stack(
            all_router_logits)  # of shape (num_hidden_layers, num_tokens, num_local_experts)
        selected_experts = torch.topk(all_router_logits, 2, dim=-1)[1].reshape(
            config.num_hidden_layers, -1
        )  # of shape (num_hidden_layers, num_tokens * 2)
        for layer_idx in range(config.num_hidden_layers):
            unique, counts = torch.unique(selected_experts[layer_idx], return_counts=True)
            expert_routed_distribution[layer_idx, unique.cpu()] += counts.cpu()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(expert_routed_distribution, os.path.join(save_dir, f"routing-count.pt"))


if __name__ == "__main__":
    Fire(dump_mixtral_routing_top_trace)
