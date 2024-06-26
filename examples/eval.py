import os

import sys
print(sys.path)

import argparse
from lm_eval import evaluator
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralForCausalLM

from transformers import AutoTokenizer
from awq.evaluation import (
    evaluate_perplexity,
    eval_librispeech,
    eval_mmlu,
    eval_humaneval,
    eval_kl_divergence,
)

def run_eval(
        model_path, quant_file, device, tasks, task_batch_size, task_n_shot,
        task_use_pretrained, pretrained_safetensors, args
    ):
    """
    Post quantization: Evaluate perplexity on wikitext with EleutherAI Evaluation Harness
    """
    tasks = tasks.split(',')

    # Load model
    if len(tasks) == 1 and tasks[0] != "mmlu" and tasks[0] != "librispeech":
        if not args.is_quantized:
            print(f'Loading original model from {model_path}')
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        else: 
            if task_use_pretrained:
                model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=pretrained_safetensors)
            else:
                model = AutoAWQForCausalLM.from_quantized(model_path, quant_file, fuse_layers=False)

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load adapter
    if len(tasks) == 1 and tasks[0] == 'wikitext':
        if args.model_path == 'mistralai/Mixtral-8x7B-v0.1':
            average_ppl = evaluate_perplexity(model, tokenizer)
        else:
            average_ppl = evaluate_perplexity(model.model, tokenizer)
            
    elif len(tasks) == 1 and tasks[0] == 'librispeech':
        eval_librispeech(model_path)
    
    elif len(tasks) == 1 and tasks[0] == 'mmlu':
        eval_mmlu(model_path, task_n_shot, task_batch_size, device, task_use_pretrained)
    
    elif len(tasks) == 1 and tasks[0] == 'humaneval':
        eval_humaneval(model, tokenizer)
    
    elif len(tasks) == 1 and tasks[0] == 'kldiv':
        eval_kl_divergence(model.model, model.model, tokenizer, seqlen=1024)

    else:
        # Evaluate perplexity of quantized model
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            batch_size=task_batch_size,
            no_cache=True,
            num_fewshot=task_n_shot,
        )

        print(evaluator.make_table(results))


    ppl_log = f'{args.model_path} |'
    
    file_path = 'awq_perplexity_results.txt'  # Specify your desired file path here
    with open(file_path, 'a') as file:  # Open file in append mode
        file.write(f'{ppl_log} | {average_ppl}\n')  # Write all_perplexity to the file, followed by a newline

    print(f'Perplexity value {average_ppl} has been added to {file_path}.')

if __name__ == '__main__':
    """
    - Run perplexity of quantized model:
    python examples/eval.py --model_path casperhansen/mistral-7b-instruct-v0.1-awq

    - Run perplexity unquantized FP16 model:
    python examples/eval.py --use_pretrained --model_path lmsys/vicuna-7b-v1.5

    - Run MMLU of quantized model:
    python examples/eval.py --model_path TheBloke/zephyr-7B-beta-AWQ --tasks mmlu --n_shot 1 --batch_size 4
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to hf model')
    parser.add_argument('--quant_file', default='', type=str, help='Path to quantized AWQ model file')
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument('--device', type=str, default='cuda:7', help='Device to load model to')
    parser.add_argument("--use_pretrained", default=False, action='store_true',
                        help="Pass '--use_pretrained' to use a pretrained model running FP16")
    parser.add_argument("--pretrained_safetensors", default=False, action='store_true',
                        help="Load safetensors for FP16 model")
    parser.add_argument('--tasks', type=str, default='wikitext', help='Tasks to evaluate. '
                    'Separate tasks by comma for multiple tasks.'
                    'https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_shot', type=int, default=0)
    args = parser.parse_args()

    run_eval(
        args.model_path, args.quant_file, args.device,
        args.tasks, args.batch_size, args.n_shot, args.use_pretrained,
        args.pretrained_safetensors, args
    )
    

# python examples/eval.py --model_path quantized_mistral-instruct-v0.2-awq-w_bit.2-group_size.64 --is_quantized

# python examples/eval.py --model_path quantized_mistral-instruct-v0.2-awq-w_bit.4-group_size.64  --is_quantized

# python examples/eval.py --model_path quantized_mistral-instruct-v0.2-awq-w_bit.8-group_size.64 --is_quantized

# python examples/eval.py --model_path quantized_mistral-instruct-v0.2-awq-w_bit.2-group_size.128 --is_quantized

# python examples/eval.py --model_path quantized_mistral-instruct-v0.2-awq-w_bit.4-group_size.128 --is_quantized

# python examples/eval.py --model_path mistralai/Mixtral-8x7B-v0.1