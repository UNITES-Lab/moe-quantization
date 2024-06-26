import sys
sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
print(sys.path)


from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model, GPTQQuantizer_deepseek
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer, TextGenerationPipeline
import logging
from datasets import load_dataset

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision
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


def moe_quantize_config(args):
    if args.bits == 'all_4':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }
        
        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
        
    if args.bits == 'all_2':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 2

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 2, 
            'model.layers.0.self_attn.k_proj': 2,
            'model.layers.0.self_attn.v_proj': 2,
            'model.layers.0.self_attn.o_proj': 2,
            'model.layers.0.mlp.gate_proj': 2,
            'model.layers.0.mlp.up_proj': 2,
            'model.layers.0.mlp.down_proj': 2
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == 'all_8':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 8

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 8

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    if args.bits == 'moe.all_mlp.2+other_block.4':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    if args.bits == 'moe.shared_4.other.2+other_block_4':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    if args.bits == "moe.shared_2.other.4+other_block_4":
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 4

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 4, 
            'model.layers.0.self_attn.k_proj': 4,
            'model.layers.0.self_attn.v_proj': 4,
            'model.layers.0.self_attn.o_proj': 4,
            'model.layers.0.mlp.gate_proj': 4,
            'model.layers.0.mlp.up_proj': 4,
            'model.layers.0.mlp.down_proj': 4
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == "moe.all_mlp.4+other_block.8":
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == 'moe.shared_4.other.2+other_block.8':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 4

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 2

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit
    
    if args.bits == 'moe.shared_2.other.4+other_block.8':
        moe_block_bit_dict = {}

        for i in range(4):
            key = f"self_attn.{['q_proj', 'k_proj', 'v_proj', 'o_proj'][i]}"
            moe_block_bit_dict[key] = 8

        for i in range(64):
            for part in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"mlp.experts.{i}.{part}"
                moe_block_bit_dict[key] = 2

        for part in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"mlp.shared_experts.{part}"
            moe_block_bit_dict[key] = 4

        deeepseek_bit = {
            'model.layers.0.self_attn.q_proj': 8, 
            'model.layers.0.self_attn.k_proj': 8,
            'model.layers.0.self_attn.v_proj': 8,
            'model.layers.0.self_attn.o_proj': 8,
            'model.layers.0.mlp.gate_proj': 8,
            'model.layers.0.mlp.up_proj': 8,
            'model.layers.0.mlp.down_proj': 8
        }

        for block_num in range(1, 28):
            for layer in moe_block_bit_dict:
                key = f'model.layers.{block_num}' + '.' + layer
                deeepseek_bit[key] = moe_block_bit_dict[layer]
        return deeepseek_bit

    
    raise ValueError("Invalid bits")

def main():
    parser = ArgumentParser()
    parser.add_argument("--bits", type=str)
    
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--quantized_model_file_base_name", type=str, default=None)
    parser.add_argument("--quant_path", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=128)    
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"/home/LeiFeng/pingzhi/moe_quantize/quantize_gptq_deepseek_{args.bits}.log"
    )
        

    args_dict = vars(args)
    logging.info("Command-line arguments: %s", args_dict)


    model_name = args.model_name    
    quant_path = f'autogptq_{model_name}-gptq_w_bit_{args.bits}'
    quantized_model_file_base_name = f'{model_name.split("/")[-1]}-gptq_w_bit_{args.bits}'
    
    deeepseek_bit = moe_quantize_config(args)

    quantize_config = BaseQuantizeConfig_mixed_precision(
        bits=deeepseek_bit,  # quantize model to 4-bit
        group_size=args.group_size,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        model_file_base_name = quantized_model_file_base_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoGPTQForCausalLM_mixed_precision.from_pretrained(model_name, quantize_config, torch_dtype=torch.float16, trust_remote_code=True)
    
    quantization_dataset = get_wikitext2(tokenizer=tokenizer, seqlen=4096, nsamples=args.nsamples, split="train")
    logging.info(f"Quantization dataset loaded with {args.nsamples} samples")
    logging.info(f"Quantizing model to {args.bits}-bit")
    logging.info(f"Quantization config: {deeepseek_bit}")
    logging.info(f"Quantized begin!!!!")
    model.quantize(quantization_dataset)
    logging.info(f"Quantized finish!!!!")
    
    logging.info(f"Quantized model begin to save")
    model.save_quantized(quant_path)
    logging.info(f"Quantized model saved to {quant_path}")



if __name__ == "__main__":

    main()