# sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
# print(sys.path)

import argparse
import json
import os

from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM_mixed_precision
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

LM_EVAL_TASK_KWARGS_DICT = {
    "winogrande": {"task": "winogrande", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "copa": {"task": "copa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "openbookqa": {"task": "openbookqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    # "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "rte": {"task": "rte", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 16, "metric": "acc"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default='deepseek-ai/deepseek-moe-16b-chat')
    parser.add_argument("--quant_model_path", type=str)

    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text.",
    )
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument(
        "--disable_exllama",
        action="store_true",
        help="Whether to use disable exllama kernel",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Whether to skip MMLU",
    )
    args = parser.parse_args()

    if args.is_quantized:
        quantized_model_file_base_name = args.quant_model_path.split("/")[-1]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
            args.quant_model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            model_basename=quantized_model_file_base_name,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_mlp=False,
            inject_fused_attention=False,
            # disable_exllama=args.disable_exllama,
        )

    save_file_path = os.path.join(f"{args.quant_model_path.split('/')[0]}",
                                  f"eval_result_{args.quant_model_path.split('/')[-1]}.json")
    all_metrics = {}
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            all_metrics = json.load(file)

    if args.proxy:
        LM_EVAL_TASK_KWARGS_DICT.pop("mmlu")
        print("Skip MMLU for proxy benchmark as it is too large.")

    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        print(f"Evaluating task: {task_kwargs['task']}")
        task_name = task_kwargs["task"]
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=task_kwargs["batch_size"],
        )
        initialize_tasks(verbosity="ERROR")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_name,
            num_fewshot=task_kwargs["num_fewshot"],
            batch_size=task_kwargs["batch_size"],
            log_samples=False,
        )
        metric = task_kwargs["metric"]
        for key, value in results["results"][task_name].items():
            if key.startswith(metric + ","):
                all_metrics[f"{task_name}_{metric}"] = value

        with open(save_file_path, 'w') as file:
            json.dump(all_metrics, file, indent=4)

    print(">>>>> Results <<<<<")
    if args.is_quantized:
        print(f"Quantization on {args.model_name} from {args.quant_model_path}")
    else:
        print(f"No quantization on {args.model_name}")
    average = sum(v for v in all_metrics.values()) / len(all_metrics)
    all_metrics["average"] = average
    print(f"Metrics: {all_metrics}")
