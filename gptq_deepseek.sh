
#!/bin/bash

# conda activate autoawq


export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export DEBUG=0
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits all_4 



export DEBUG=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits all_2


export DEBUG=0
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.all_mlp.2+other_block.4



export DEBUG=0
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_4.other.2+other_block_4



export DEBUG=0
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_2.other.4+other_block_4



export DEBUG=0
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits all_8




export DEBUG=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.all_mlp.4+other_block.8



export DEBUG=0
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_4.other.2+other_block.8


export DEBUG=0
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH
python quantize_gptq_deepseek.py \
    --model_name deepseek-ai/deepseek-moe-16b-chat \
    --nsamples 512 \
    --group_size 64 \
    --bits moe.shared_2.other.4+other_block.8