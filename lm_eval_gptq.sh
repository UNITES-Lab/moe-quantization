export DEBUG=0
export PYTHONPATH=/home/LeiFeng/pingzhi/moe_quantize/optimum/:$PYTHONPATH:/home/LeiFeng/pingzhi/moe_quantize/auto_gptq/:$PYTHONPATH

#for bits in 4 2
#do
#CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
#    --model_name mistralai/Mixtral-8x7B-v0.1 \
#    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_$bits.most_used \
#    --bits $bits \
#    --is_quantized
#done
DEBUG=0 CUDA_VISIBLE_DEVICES=0,1,2 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random4_per_layer_seed43 \
    --is_quantized

DEBUG=0 CUDA_VISIBLE_DEVICES=3,4,5 python lm_eval_gptq.py \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --quant_model_path autogptq_mistralai/Mixtral-8x7B-v0.1-gptq_w_bit_main_2.attn_4.random4_per_layer_seed44 \
    --is_quantized