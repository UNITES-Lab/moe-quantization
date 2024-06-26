
#!/bin/bash
export DEBUG=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_all_4 \
    --bits all_4 \
    --is_quantized



export DEBUG=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_all_2 \
    --bits all_2 \
    --is_quantized \




export DEBUG=0
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.all_mlp.2+other_block.4 \
    --bits moe.all_mlp.2+other_block.4 \
    --is_quantized \



export DEBUG=0
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_4.other.2+other_block_4 \
    --bits moe.shared_4.other.2+other_block_4 \
    --is_quantized \



export DEBUG=0
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_2.other.4+other_block_4 \
    --bits moe.shared_2.other.4+other_block_4 \
    --is_quantized 



export DEBUG=0
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_all_8 \
    --bits all_8 \
    --is_quantized 



export DEBUG=0
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.all_mlp.4+other_block.8 \
    --bits moe.all_mlp.4+other_block.8 \
    --is_quantized 


export DEBUG=0
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_4.other.2+other_block.8 \
    --bits moe.shared_4.other.2+other_block.8 \
    --is_quantized 



export DEBUG=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_2.other.4+other_block.8 \
    --bits moe.shared_2.other.4+other_block.8 \
    --is_quantized 




export DEBUG=0
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek_eval.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --quant_model_path autogptq_quantize_model/deepseek-moe-16b-chat-gptq_w_bit_moe.shared_2.other.4+other_block.8 \
    --bits moe.shared_2.other.4+other_block.8 \
    --is_quantized 

