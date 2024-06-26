import sys

sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
print(sys.path)
import re
import torch

from auto_gptq import (AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision)
from fire import Fire


def _trial_loop(model, mixtral_bits, expert_add_bits, target_average_bits):
    num_experts_add_bits = 0
    for block_num in range(0, 32):
        for expert_id in range(0, 8):
            if all(mixtral_bits[
                       f"model.layers.{block_num}.block_sparse_moe.experts.{expert_id}.{part}"] < expert_add_bits
                   for part in ['w1', 'w2', 'w3']):
                for part in ['w1', 'w2', 'w3']:
                    mixtral_bits[
                        f"model.layers.{block_num}.block_sparse_moe.experts.{expert_id}.{part}"] = expert_add_bits
                num_experts_add_bits += 1
                if _compute_average_bit(model, mixtral_bits) >= target_average_bits:
                    return num_experts_add_bits, mixtral_bits

    return num_experts_add_bits, mixtral_bits


def _compute_average_bit(model, bits_dict):
    total_bits = 0
    total_num_params = 0
    for name, module in model.model.named_modules():
        if name not in bits_dict:
            continue
        bits = bits_dict[name]
        num_params = sum(p.numel() for p in module.parameters())
        total_bits += bits * num_params
        total_num_params += num_params

    average_bits = total_bits / total_num_params
    return average_bits


def calculate_mixtral_num_experts_to_add_bits(
        bits_config_str: str, expert_add_bits: int, target_average_bits: int
):
    mixtral_bits = dict()
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
            mixtral_bits[key] = moe_block_bit_dict[layer]
    # Special expert bits, e.g. "exp_l1e3_16": 16-bit for expert 3 in layer 1
    special_expert_bits = re.findall(r"exp_l(\d+)e(\d+)_(\d+)", bits_config_str)
    for layer, expert, bits in special_expert_bits:
        for part in ['w1', 'w2', 'w3']:
            key = f"model.layers.{int(layer)}.block_sparse_moe.experts.{int(expert)}.{part}"
            mixtral_bits[key] = int(bits)

    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits={k: v for k, v in mixtral_bits.items() if v != 16},  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        model_file_base_name="tmp_model",  # the base name of the quantized model file
    )

    model = AutoGPTQForCausalLM_mixed_precision.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", quantize_config,
        torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    before_bits = _compute_average_bit(model, mixtral_bits)

    # Calculate the average bit-width of the model
    num_experts_add_bits, mixtral_bits = _trial_loop(model, mixtral_bits, expert_add_bits, target_average_bits)

    print("=====================================")
    print(f"Bits config: {bits_config_str}")
    print(f"Number of experts to add bits: {num_experts_add_bits}")
    print(f"Before bits: {before_bits:.2f}")
    print(f"After bits: {_compute_average_bit(model, mixtral_bits):.2f}")


if __name__ == "__main__":
    Fire(calculate_mixtral_num_experts_to_add_bits)
