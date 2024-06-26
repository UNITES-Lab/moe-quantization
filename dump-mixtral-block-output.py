# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/29
import os.path
import random

import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralDecoderLayer

random.seed(233)


def _get_profiling_dataset(tokenizer, seq_len, num_samples):
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([" \n" if s == "" else s for s in data["text"][-1000:]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


@torch.no_grad()
def collect_mixtral_block_output(
        seq_len=1024,
        num_samples=400,
        save_dir="/data/data12/pingzhi/data/wikitext_block_output"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    block_outputs = {}

    def _custom_ffn_forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_output = self._original_forward(hidden_states, *args, **kwargs)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_outputs[self._module_name].append(output_token)
        return original_output

    for name, module in model.named_modules():
        if isinstance(module, MixtralDecoderLayer):
            block_outputs[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()
    dataset = _get_profiling_dataset(tokenizer, seq_len, num_samples)

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for key, pairs in block_outputs.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


if __name__ == "__main__":
    Fire(collect_mixtral_block_output)
