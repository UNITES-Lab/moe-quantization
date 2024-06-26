#!/bin/bash

# Ensure script stops if an error occurs
set -e

# Export environment variables common to all commands
export DEBUG=0
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH


python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_all_4 \
    --bits all_4 \
    --is_quantized

python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_all_2 \
    --bits all_2 \
    --is_quantized

python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.all_mlp.2+other_block.4 \
    --bits moe.all_mlp.2+other_block.4 \
    --is_quantized

python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_4.other.2+other_block_4 \
    --bits moe.shared_4.other.2+other_block_4 \
    --is_quantized


python lm_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_2.other.4+other_block_4 \
    --bits moe.shared_2.other.4+other_block_4 \
    --is_quantized 