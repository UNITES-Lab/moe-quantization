import sys

# sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
print(sys.path)
import re
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer
from datasets import load_dataset
from auto_gptq import (
    AutoGPTQForCausalLM_mixed_precision,
    BaseQuantizeConfig_mixed_precision
)
import logging


def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
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


def mixtral_quantize_config(bits_config_str: str):
    mixtral_bit = dict()
    # The main weight bits
    main_bits = re.search(r"main_(\d)", bits_config_str)
    if main_bits is None:
        raise ValueError(f"Invalid bits config string: {bits_config_str}")
    main_bits = int(main_bits.group(1))
    moe_block_bit_dict = {}
    for i in range(4):
        key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
        if "attn" in bits_config_str:
            attn_bits = re.search(r"attn_(\d)", bits_config_str)[1]
            moe_block_bit_dict[key] = int(attn_bits)
        else:
            moe_block_bit_dict[key] = main_bits
    for i in range(8):
        for part in ['w1', 'w2', 'w3']:
            key = f"block_sparse_moe.experts.{i}.{part}"
            moe_block_bit_dict[key] = main_bits
    for block_num in range(0, 32):
        for layer in moe_block_bit_dict:
            key = f'model.layers.{block_num}' + '.' + layer
            mixtral_bit[key] = moe_block_bit_dict[layer]

    # Special expert bits, e.g. "exp_l1e3_16": 16-bit for expert 3 in layer 1
    special_expert_bits = re.findall(r"exp_l(\d+)e(\d+)_(\d+)", bits_config_str)
    for layer, expert, bits in special_expert_bits:
        for part in ['w1', 'w2', 'w3']:
            key = f"model.layers.{int(layer)}.block_sparse_moe.experts.{int(expert)}.{part}"
            mixtral_bit[key] = int(bits)

    # Special layer bits, e.g. "layer_16_4": 4-bit for layer 16
    special_layer_bits = re.findall(r"layer_(\d+)_(\d+)", bits_config_str)
    for layer, bits in special_layer_bits:
        print(f"Applying {bits}-bit to layer {layer}")
        for key in mixtral_bit:
            if f"model.layers.{int(layer)}.block_sparse_moe." in key:
                mixtral_bit[key] = int(bits)

    # Special module name keywords, e.g. "keyword__gate_proj__4": 4-bit for all gate_proj modules
    special_module_bits = re.findall(r"keyword__(\w+)__(\d+)", bits_config_str)
    for module_key, bits in special_module_bits:
        print(f"Applying {bits}-bit to module {module_key}")
        for key in mixtral_bit:
            if module_key in key:
                mixtral_bit[key] = int(bits)

    return mixtral_bit


def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bits_name", type=str, default=None)
    parser.add_argument("--bits_dict_overwrite", type=str, default=None)

    args = parser.parse_args()

    bits_name = str(args.bits) if args.bits_name is None else args.bits_name

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"./quantize_gptq_mixtral_{bits_name}.log"
    )

    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)

    model_name = args.model_name
    quant_path = f'autogptq_{model_name}-gptq_w_bit_{bits_name}'
    quantized_model_file_base_name = f'{model_name.split("/")[-1]}-gptq_w_bit_{bits_name}'

    mixtral_bits = mixtral_quantize_config(args.bits)

    if args.bits_dict_overwrite is not None:
        overwrite_bits = torch.load(args.bits_dict_overwrite)
        print(f"Overwrite bits from {args.bits_dict_overwrite}")
        mixtral_bits.update(overwrite_bits)

    print("====== First 32 bits config items ======")
    for i, (k, v) in enumerate(mixtral_bits.items()):
        if i >= 32:
            break
        print(f"{k}: {v}")
    print()

    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits={k: v for k, v in mixtral_bits.items() if v != 16},  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        model_file_base_name=quantized_model_file_base_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoGPTQForCausalLM_mixed_precision.from_pretrained(
        model_name, quantize_config, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    # Calculate the average bit-width of the model
    total_bits = 0
    total_num_params = 0
    for name, module in model.model.named_modules():
        if name not in mixtral_bits:
            continue
        bits = mixtral_bits[name]
        num_params = sum(p.numel() for p in module.parameters())
        total_bits += bits * num_params
        total_num_params += num_params

    average_bits = total_bits / total_num_params
    print(f"Average bit-width of the model w/ {bits_name}: {average_bits:.2f}")

    quantization_dataset = get_wikitext2(tokenizer=tokenizer, seqlen=4096, nsamples=args.nsamples, split="train")
    model.quantize(quantization_dataset)
    model.save_quantized(quant_path)
    logging.info(f"Quantized model saved to {quant_path}")


if __name__ == "__main__":
    main()
