# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/4/29
import os.path
import random
from typing import Tuple, Optional

import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock, MixtralDecoderLayer


@torch.no_grad()
def collect_mixtral_ffn_predictor_train_data(
        seq_len=1024,
        num_samples=400,
        save_dir="/data/data4/pingzhi/data/ffn_input_output_pairs"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))
        return original_output

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralSparseMoeBlock):
            block_ffn_input_output_pair[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([" \n" if s == "" else s for s in data["text"][-2000:]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_mixtral_predictor_test_data(
        seq_len=4096,
        num_samples=128,
        save_dir="/data/data8/pingzhi/data/ffn_input_output_pairs/testset",
        dataset_name: str = "wikitext"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def _custom_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_token = hidden_states.detach().clone().cpu()
        original_output = self._original_forward(hidden_states)
        output_token = original_output[0].detach().clone().cpu()
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))
        return original_output

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralSparseMoeBlock):
            block_ffn_input_output_pair[name] = []
            module._original_forward = module.forward
            module._module_name = name
            module.forward = _custom_ffn_forward.__get__(module, type(module))

    model.eval()

    if dataset_name == "wikitext":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif dataset_name == "minipile":
        data = load_dataset("JeanKaddour/minipile", split="train")
    else:
        raise ValueError("dataset_name should be 'wikitext' or 'minipile'")
    # this is GPTQ calibration data
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_mixtral_ffn_with_residual_predictor_train_data(
        seq_len=1024,
        num_samples=400,
        save_dir="/data/data7/pingzhi/data/ffn_input_output_pairs"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def _custom_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Fully Connected
        input_token = hidden_states.detach().clone().cpu()  # added
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        output_token = hidden_states.detach().clone().cpu()  # added
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))  # added

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralDecoderLayer):
            block_ffn_input_output_pair[name] = []
            module._module_name = name
            module.forward = _custom_decoder_forward.__get__(module, type(module))

    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([" \n" if s == "" else s for s in data["text"][-2000:]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_mixtral_ffn_with_residual_predictor_test_data(
        seq_len=4096,
        num_samples=128,
        save_dir="/data/data9/pingzhi/data/ffn_input_output_pairs/testset"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def _custom_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Fully Connected
        input_token = hidden_states.detach().clone().cpu()  # added
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        output_token = hidden_states.detach().clone().cpu()  # added
        with torch.no_grad():
            block_ffn_input_output_pair[self._module_name].append((input_token, output_token))  # added

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    block_ffn_input_output_pair = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralDecoderLayer):
            block_ffn_input_output_pair[name] = []
            module._module_name = name
            module.forward = _custom_decoder_forward.__get__(module, type(module))

    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for key, pairs in block_ffn_input_output_pair.items():
        torch.save(pairs, f"{save_dir}/{key}.pt")
        print(f"Saved at {save_dir}/{key}.pt")


@torch.no_grad()
def collect_mixtral_ffn_with_residual_cosine_similarity(
        seq_len=1024,
        num_samples=400,
        save_dir="results/ffn_input_output_cosine_similarity"
):
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def _custom_decoder_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # Fully Connected
        input_token = hidden_states.detach().clone().cpu()  # added
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        output_token = hidden_states.detach().clone().cpu()  # added
        with torch.no_grad():
            block_ffn_input_output_pair_cos_sim[self._module_name].append(
                torch.nn.functional.cosine_similarity(input_token, output_token, dim=-1)
            )  # added

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    block_ffn_input_output_pair_cos_sim = {}
    for name, module in model.named_modules():
        if isinstance(module, MixtralDecoderLayer):
            block_ffn_input_output_pair_cos_sim[name] = []
            module._module_name = name
            module.forward = _custom_decoder_forward.__get__(module, type(module))

    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([" \n" if s == "" else s for s in data["text"][-2000:]])
    encoded_text = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(num_samples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded_text.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    for i, data in enumerate(tqdm(dataset)):
        with torch.no_grad():
            model(**data)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    torch.save(block_ffn_input_output_pair_cos_sim, f"{save_dir}/cosine_similarity.pt")


if __name__ == "__main__":
    Fire(collect_mixtral_ffn_with_residual_cosine_similarity)
