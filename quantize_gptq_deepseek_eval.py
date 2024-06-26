import sys
sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
# print(sys.path)


import os
os.environ['HF_HOME'] = '/home/LeiFeng/pingzhi/moe_quantize/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)


from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model, GPTQQuantizer_deepseek
import torch
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer, TextGenerationPipeline
import logging
from datasets import load_dataset

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision


import argparse
import os

import torch
from transformers import AutoTokenizer

from auto_gptq.utils import Perplexity



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
    model.quantize(quantization_dataset)

    model.save_quantized(quant_path)
    logging.info(f"Quantized model saved to {quant_path}")

def eval():
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    python examples/benchmark/perplexity.py \
        --model_name TheBloke/open-llama-7b-open-instruct-GPTQ \
        --model_basename gptq_model-4bit-128g \
        --is_quantized

    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-chat')
    parser.add_argument("--quant_model_path", type=str)
    parser.add_argument("--bits", type=str)
    
    # parser.add_argument("--model_basename", type=str, default=None, help="Model file's basename.")
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text.",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="Max memory used in each GPU.",
    )
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Whether to use safetensors model file",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument(
        "--disable_exllama",
        action="store_true",
        help="Whether to use disable exllama kernel",
    )
    args = parser.parse_args()


    if args.is_quantized:
        args.quantized_model_file_base_name = f'{args.model_name.split("/")[-1]}-gptq_w_bit_{args.bits}'
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        max_memory = {}
        if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
            if torch.cuda.is_available():
                max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
        if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
            max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
        if not max_memory:
            max_memory = None

        if args.use_safetensors:
            print(
                "The argument --use_safetensors is deprecrated and will be removed in the next release. It is now the default behavior."
            )

        model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
            args.quant_model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            model_basename=args.quantized_model_file_base_name,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_mlp=False,
            inject_fused_attention=False,
            # disable_exllama=args.disable_exllama,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    ppl = Perplexity(
        model,
        tokenizer,
        args.dataset_path,
        args.dataset_name,
        args.split,
        args.text_column,
    )
    all_perplexity = ppl.calculate_perplexity(args.n_ctx, args.n_batch)
    average_ppl = sum(all_perplexity) / len(all_perplexity)
    
    if args.is_quantized:
        ppl_log = f'{args.quant_model_path} | {args.bits}'
    else:
        ppl_log = f'{args.model_name} | full precision'
        
        
    file_path = 'perplexity_results.txt'  # Specify your desired file path here
    with open(file_path, 'a') as file:  # Open file in append mode
        file.write(f'{ppl_log} | {average_ppl}\n')  # Write all_perplexity to the file, followed by a newline

    print(f'Perplexity value {average_ppl} has been added to {file_path}.')


if __name__ == "__main__":

    eval()
    
    
    
    
# /home/LeiFeng/pingzhi/smoothquant/smoothquant/lm_eval
# cp -r /home/LeiFeng/pingzhi/smoothquant/smoothquant/lm_eval /home/LeiFeng/pingzhi/moe_quantize
# cp -r /home/LeiFeng/pingzhi/smoothquant/smoothquant/lm_eval.py /home/LeiFeng/pingzhi/moe_quantize