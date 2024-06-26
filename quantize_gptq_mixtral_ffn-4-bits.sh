# Attention 4 bits + Left Frequency 4 bits (126 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l12e1_4.exp_l14e4_4.exp_l24e2_4.exp_l6e5_4.exp_l23e1_4.exp_l2e0_4.exp_l8e5_4.exp_l11e1_4.exp_l8e0_4.exp_l1e4_4.exp_l10e1_4.exp_l12e2_4.exp_l14e5_4.exp_l20e6_4.exp_l4e3_4.exp_l17e2_4.exp_l11e4_4.exp_l12e0_4.exp_l16e7_4.exp_l1e3_4.exp_l31e7_4.exp_l29e2_4.exp_l9e0_4.exp_l11e2_4.exp_l10e2_4.exp_l5e1_4.exp_l14e3_4.exp_l7e5_4.exp_l23e0_4.exp_l15e1_4.exp_l10e0_4.exp_l22e0_4.exp_l13e7_4.exp_l2e1_4.exp_l31e3_4.exp_l28e1_4.exp_l4e2_4.exp_l10e3_4.exp_l23e6_4.exp_l15e2_4.exp_l17e5_4.exp_l21e6_4.exp_l19e0_4.exp_l30e7_4.exp_l30e4_4.exp_l28e3_4.exp_l9e3_4.exp_l3e2_4.exp_l12e3_4.exp_l15e0_4.exp_l15e7_4.exp_l16e6_4.exp_l7e6_4.exp_l25e1_4.exp_l11e0_4.exp_l18e0_4.exp_l12e5_4.exp_l17e7_4.exp_l22e3_4.exp_l30e2_4.exp_l0e2_4.exp_l27e5_4.exp_l19e4_4.exp_l11e5_4.exp_l5e2_4.exp_l26e3_4.exp_l25e5_4.exp_l17e1_4.exp_l25e4_4.exp_l3e5_4.exp_l14e1_4.exp_l24e1_4.exp_l21e3_4.exp_l26e2_4.exp_l4e7_4.exp_l9e6_4.exp_l9e5_4.exp_l6e0_4.exp_l21e0_4.exp_l31e2_4.exp_l6e7_4.exp_l10e7_4.exp_l0e7_4.exp_l3e6_4.exp_l27e0_4.exp_l11e6_4.exp_l20e7_4.exp_l1e0_4.exp_l19e5_4.exp_l20e2_4.exp_l29e6_4.exp_l13e4_4.exp_l8e6_4.exp_l29e7_4.exp_l23e7_4.exp_l19e2_4.exp_l21e7_4.exp_l12e4_4.exp_l13e6_4.exp_l15e4_4.exp_l5e5_4.exp_l6e6_4.exp_l26e6_4.exp_l27e2_4.exp_l7e4_4.exp_l25e3_4.exp_l6e2_4.exp_l29e1_4.exp_l2e6_4.exp_l15e6_4.exp_l5e0_4.exp_l22e5_4.exp_l26e4_4.exp_l20e5_4.exp_l18e2_4.exp_l16e5_4.exp_l7e0_4.exp_l30e1_4.exp_l11e7_4.exp_l31e5_4.exp_l2e5_4.exp_l16e4_4.exp_l17e0_4.exp_l26e1_4.exp_l29e3_4 \
      --bits_name main_2.attn_4.frequency_avg_3bits

# Attention 4 bits + Left Task Specific 4 bits (126 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l30e7_4.exp_l31e3_4.exp_l31e1_4.exp_l29e7_4.exp_l28e4_4.exp_l8e0_4.exp_l23e1_4.exp_l27e2_4.exp_l2e0_4.exp_l24e1_4.exp_l12e0_4.exp_l6e5_4.exp_l26e4_4.exp_l12e1_4.exp_l11e1_4.exp_l21e0_4.exp_l17e1_4.exp_l31e7_4.exp_l20e6_4.exp_l29e6_4.exp_l15e2_4.exp_l31e2_4.exp_l0e5_4.exp_l19e0_4.exp_l0e2_4.exp_l11e4_4.exp_l4e3_4.exp_l18e2_4.exp_l16e3_4.exp_l26e3_4.exp_l3e2_4.exp_l16e7_4.exp_l24e2_4.exp_l17e5_4.exp_l10e1_4.exp_l23e0_4.exp_l2e1_4.exp_l13e4_4.exp_l1e4_4.exp_l16e6_4.exp_l27e5_4.exp_l14e5_4.exp_l4e2_4.exp_l20e2_4.exp_l27e0_4.exp_l25e4_4.exp_l29e2_4.exp_l14e4_4.exp_l15e1_4.exp_l9e3_4.exp_l21e3_4.exp_l10e2_4.exp_l17e0_4.exp_l19e4_4.exp_l28e7_4.exp_l14e1_4.exp_l10e0_4.exp_l22e0_4.exp_l9e0_4.exp_l17e2_4.exp_l12e2_4.exp_l13e7_4.exp_l5e0_4.exp_l23e6_4.exp_l15e7_4.exp_l30e1_4.exp_l20e7_4.exp_l8e5_4.exp_l25e3_4.exp_l26e1_4.exp_l11e5_4.exp_l14e3_4.exp_l6e2_4.exp_l7e6_4.exp_l21e6_4.exp_l3e5_4.exp_l17e7_4.exp_l26e2_4.exp_l22e7_4.exp_l2e6_4.exp_l25e0_4.exp_l22e3_4.exp_l5e5_4.exp_l12e4_4.exp_l11e6_4.exp_l5e1_4.exp_l7e4_4.exp_l8e6_4.exp_l0e3_4.exp_l18e3_4.exp_l6e0_4.exp_l25e1_4.exp_l4e7_4.exp_l12e6_4.exp_l6e7_4.exp_l1e3_4.exp_l12e3_4.exp_l24e5_4.exp_l19e2_4.exp_l15e0_4.exp_l13e6_4.exp_l20e5_4.exp_l7e0_4.exp_l0e7_4.exp_l15e4_4.exp_l21e7_4.exp_l19e5_4.exp_l13e5_4.exp_l11e0_4.exp_l5e2_4.exp_l10e7_4.exp_l15e6_4.exp_l9e1_4.exp_l29e4_4.exp_l10e5_4.exp_l0e1_4.exp_l11e2_4.exp_l27e4_4.exp_l19e7_4.exp_l16e4_4.exp_l28e2_4.exp_l3e6_4.exp_l12e5_4.exp_l28e1_4.exp_l3e3_4.exp_l9e5_4 \
      --bits_name main_2.attn_4.task_specific_avg_3bits

# Attention 4 bits + Left WANDA 4 bits (126 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l31e1_4.exp_l31e5_4.exp_l30e7_4.exp_l30e5_4.exp_l30e3_4.exp_l29e7_4.exp_l29e2_4.exp_l29e1_4.exp_l28e0_4.exp_l31e7_4.exp_l28e3_4.exp_l29e6_4.exp_l30e2_4.exp_l29e0_4.exp_l28e1_4.exp_l29e3_4.exp_l28e5_4.exp_l31e2_4.exp_l28e4_4.exp_l28e6_4.exp_l30e4_4.exp_l28e2_4.exp_l29e4_4.exp_l28e7_4.exp_l27e0_4.exp_l27e5_4.exp_l27e4_4.exp_l26e6_4.exp_l30e0_4.exp_l27e1_4.exp_l27e6_4.exp_l27e2_4.exp_l30e6_4.exp_l26e1_4.exp_l31e3_4.exp_l26e2_4.exp_l26e3_4.exp_l27e3_4.exp_l25e5_4.exp_l27e7_4.exp_l30e1_4.exp_l26e4_4.exp_l29e5_4.exp_l25e4_4.exp_l26e0_4.exp_l26e7_4.exp_l25e3_4.exp_l25e1_4.exp_l23e1_4.exp_l31e0_4.exp_l24e2_4.exp_l24e1_4.exp_l25e7_4.exp_l25e2_4.exp_l24e6_4.exp_l25e0_4.exp_l25e6_4.exp_l24e5_4.exp_l26e5_4.exp_l24e4_4.exp_l31e6_4.exp_l31e4_4.exp_l23e0_4.exp_l23e6_4.exp_l24e3_4.exp_l24e0_4.exp_l23e7_4.exp_l24e7_4.exp_l22e3_4.exp_l22e0_4.exp_l23e2_4.exp_l22e7_4.exp_l23e4_4.exp_l22e5_4.exp_l21e3_4.exp_l21e6_4.exp_l23e3_4.exp_l21e0_4.exp_l22e2_4.exp_l22e4_4.exp_l21e7_4.exp_l22e1_4.exp_l22e6_4.exp_l20e6_4.exp_l23e5_4.exp_l21e2_4.exp_l21e5_4.exp_l20e7_4.exp_l20e2_4.exp_l21e1_4.exp_l21e4_4.exp_l20e4_4.exp_l20e1_4.exp_l19e0_4.exp_l20e5_4.exp_l19e5_4.exp_l20e3_4.exp_l19e2_4.exp_l19e4_4.exp_l19e6_4.exp_l20e0_4.exp_l19e7_4.exp_l19e3_4.exp_l18e2_4.exp_l18e4_4.exp_l18e7_4.exp_l18e1_4.exp_l19e1_4.exp_l18e0_4.exp_l18e6_4.exp_l18e3_4.exp_l17e2_4.exp_l18e5_4.exp_l17e1_4.exp_l17e5_4.exp_l17e0_4.exp_l17e7_4.exp_l16e7_4.exp_l17e4_4.exp_l16e6_4.exp_l17e6_4.exp_l16e3_4.exp_l16e4_4.exp_l17e3_4.exp_l16e1_4.exp_l16e5_4 \
      --bits_name main_2.attn_4.wanda_avg_3bits

# Attention 4 bits + Left Massive Token Expert 4 bits (126 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l19e6_4.exp_l23e3_4.exp_l27e4_4.exp_l23e1_4.exp_l15e3_4.exp_l15e7_4.exp_l16e0_4.exp_l11e5_4.exp_l9e7_4.exp_l6e4_4.exp_l1e3_4.exp_l28e2_4.exp_l27e1_4.exp_l22e3_4.exp_l16e3_4.exp_l18e0_4.exp_l20e3_4.exp_l11e3_4.exp_l18e7_4.exp_l13e0_4.exp_l21e6_4.exp_l17e6_4.exp_l14e2_4.exp_l10e4_4.exp_l4e5_4.exp_l5e6_4.exp_l2e0_4.exp_l3e4_4.exp_l25e7_4.exp_l12e6_4.exp_l14e6_4.exp_l26e0_4.exp_l24e2_4.exp_l12e2_4.exp_l24e7_4.exp_l13e4_4.exp_l21e7_4.exp_l26e7_4.exp_l25e2_4.exp_l17e4_4.exp_l20e6_4.exp_l10e2_4.exp_l9e3_4.exp_l8e1_4.exp_l6e1_4.exp_l7e2_4.exp_l5e0_4.exp_l3e5_4.exp_l8e6_4.exp_l7e0_4.exp_l4e4_4.exp_l28e5_4.exp_l2e2_4.exp_l29e3_4.exp_l19e7_4.exp_l21e0_4.exp_l24e6_4.exp_l26e3_4.exp_l25e6_4.exp_l22e1_4.exp_l26e4_4.exp_l21e2_4.exp_l25e1_4.exp_l20e0_4.exp_l2e7_4.exp_l13e1_4.exp_l14e5_4.exp_l12e7_4.exp_l17e7_4.exp_l12e0_4.exp_l13e3_4.exp_l14e7_4.exp_l10e6_4.exp_l16e1_4.exp_l24e3_4.exp_l18e2_4.exp_l18e6_4.exp_l10e3_4.exp_l7e3_4.exp_l3e1_4.exp_l5e5_4.exp_l8e4_4.exp_l8e7_4.exp_l4e7_4.exp_l4e3_4.exp_l3e6_4.exp_l6e3_4.exp_l5e7_4.exp_l2e4_4.exp_l7e5_4.exp_l22e2_4.exp_l29e1_4.exp_l11e6_4.exp_l28e4_4.exp_l9e4_4.exp_l22e6_4.exp_l1e5_4.exp_l27e5_4.exp_l17e3_4.exp_l1e0_4.exp_l1e2_4.exp_l20e4_4.exp_l19e5_4.exp_l19e3_4.exp_l17e1_4.exp_l20e5_4.exp_l22e4_4.exp_l1e1_4.exp_l20e7_4.exp_l27e2_4.exp_l29e0_4.exp_l19e0_4.exp_l22e0_4.exp_l11e1_4.exp_l9e6_4.exp_l19e1_4.exp_l17e5_4.exp_l9e1_4.exp_l23e5_4.exp_l19e2_4.exp_l24e1_4.exp_l1e4_4.exp_l24e5_4.exp_l1e7_4.exp_l4e2_4.exp_l25e3_4 \
      --bits_name main_2.attn_4.mass_token_avg_3bits

# Attention 4 bits + Each Layer top-4 Expert 4 bits (128 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l0e2_4.exp_l0e7_4.exp_l0e0_4.exp_l1e4_4.exp_l1e3_4.exp_l1e0_4.exp_l1e5_4.exp_l2e0_4.exp_l2e1_4.exp_l2e6_4.exp_l2e5_4.exp_l3e2_4.exp_l3e5_4.exp_l3e6_4.exp_l3e3_4.exp_l4e3_4.exp_l4e2_4.exp_l4e7_4.exp_l4e1_4.exp_l5e1_4.exp_l5e2_4.exp_l5e5_4.exp_l5e0_4.exp_l6e5_4.exp_l6e0_4.exp_l6e7_4.exp_l6e6_4.exp_l7e5_4.exp_l7e6_4.exp_l7e4_4.exp_l7e0_4.exp_l8e5_4.exp_l8e0_4.exp_l8e6_4.exp_l8e4_4.exp_l9e0_4.exp_l9e3_4.exp_l9e6_4.exp_l9e5_4.exp_l10e1_4.exp_l10e2_4.exp_l10e0_4.exp_l10e3_4.exp_l11e1_4.exp_l11e4_4.exp_l11e2_4.exp_l11e0_4.exp_l12e1_4.exp_l12e2_4.exp_l12e0_4.exp_l12e3_4.exp_l13e7_4.exp_l13e4_4.exp_l13e6_4.exp_l13e5_4.exp_l14e4_4.exp_l14e5_4.exp_l14e3_4.exp_l14e1_4.exp_l15e1_4.exp_l15e2_4.exp_l15e0_4.exp_l15e7_4.exp_l16e7_4.exp_l16e6_4.exp_l16e5_4.exp_l16e4_4.exp_l17e2_4.exp_l17e5_4.exp_l17e7_4.exp_l17e1_4.exp_l18e0_4.exp_l18e2_4.exp_l18e1_4.exp_l18e4_4.exp_l19e0_4.exp_l19e4_4.exp_l19e5_4.exp_l19e2_4.exp_l20e6_4.exp_l20e7_4.exp_l20e2_4.exp_l20e5_4.exp_l21e6_4.exp_l21e3_4.exp_l21e0_4.exp_l21e7_4.exp_l22e0_4.exp_l22e3_4.exp_l22e5_4.exp_l22e7_4.exp_l23e1_4.exp_l23e0_4.exp_l23e6_4.exp_l23e7_4.exp_l24e2_4.exp_l24e1_4.exp_l24e5_4.exp_l24e4_4.exp_l25e1_4.exp_l25e5_4.exp_l25e4_4.exp_l25e3_4.exp_l26e3_4.exp_l26e2_4.exp_l26e6_4.exp_l26e4_4.exp_l27e5_4.exp_l27e0_4.exp_l27e2_4.exp_l27e4_4.exp_l28e1_4.exp_l28e3_4.exp_l28e4_4.exp_l28e0_4.exp_l29e2_4.exp_l29e6_4.exp_l29e7_4.exp_l29e1_4.exp_l30e7_4.exp_l30e4_4.exp_l30e2_4.exp_l30e1_4.exp_l31e7_4.exp_l31e3_4.exp_l31e2_4.exp_l31e5_4 \
      --bits_name main_2.attn_4.each_layer_top4

# Attention 4 bits + Each Layer top-3 Expert 4 bits (96 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l0e2_4.exp_l0e7_4.exp_l1e4_4.exp_l1e3_4.exp_l1e0_4.exp_l2e0_4.exp_l2e1_4.exp_l2e6_4.exp_l3e2_4.exp_l3e5_4.exp_l3e6_4.exp_l4e3_4.exp_l4e2_4.exp_l4e7_4.exp_l5e1_4.exp_l5e2_4.exp_l5e5_4.exp_l6e5_4.exp_l6e0_4.exp_l6e7_4.exp_l7e5_4.exp_l7e6_4.exp_l7e4_4.exp_l8e5_4.exp_l8e0_4.exp_l8e6_4.exp_l9e0_4.exp_l9e3_4.exp_l9e6_4.exp_l10e1_4.exp_l10e2_4.exp_l10e0_4.exp_l11e1_4.exp_l11e4_4.exp_l11e2_4.exp_l12e1_4.exp_l12e2_4.exp_l12e0_4.exp_l13e7_4.exp_l13e4_4.exp_l13e6_4.exp_l14e4_4.exp_l14e5_4.exp_l14e3_4.exp_l15e1_4.exp_l15e2_4.exp_l15e0_4.exp_l16e7_4.exp_l16e6_4.exp_l16e5_4.exp_l17e2_4.exp_l17e5_4.exp_l17e7_4.exp_l18e0_4.exp_l18e2_4.exp_l18e1_4.exp_l19e0_4.exp_l19e4_4.exp_l19e5_4.exp_l20e6_4.exp_l20e7_4.exp_l20e2_4.exp_l21e6_4.exp_l21e3_4.exp_l21e0_4.exp_l22e0_4.exp_l22e3_4.exp_l22e5_4.exp_l23e1_4.exp_l23e0_4.exp_l23e6_4.exp_l24e2_4.exp_l24e1_4.exp_l24e5_4.exp_l25e1_4.exp_l25e5_4.exp_l25e4_4.exp_l26e3_4.exp_l26e2_4.exp_l26e6_4.exp_l27e5_4.exp_l27e0_4.exp_l27e2_4.exp_l28e1_4.exp_l28e3_4.exp_l28e4_4.exp_l29e2_4.exp_l29e6_4.exp_l29e7_4.exp_l30e7_4.exp_l30e4_4.exp_l30e2_4.exp_l31e7_4.exp_l31e3_4.exp_l31e2_4 \
      --bits_name main_2.attn_4.each_layer_top3

# Attention 4 bits + Each Layer top-2 Expert 4 bits (64 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l0e2_4.exp_l1e4_4.exp_l1e3_4.exp_l2e0_4.exp_l2e1_4.exp_l3e2_4.exp_l3e5_4.exp_l4e3_4.exp_l4e2_4.exp_l5e1_4.exp_l5e2_4.exp_l6e5_4.exp_l6e0_4.exp_l7e5_4.exp_l7e6_4.exp_l8e5_4.exp_l8e0_4.exp_l9e0_4.exp_l9e3_4.exp_l10e1_4.exp_l10e2_4.exp_l11e1_4.exp_l11e4_4.exp_l12e1_4.exp_l12e2_4.exp_l13e7_4.exp_l13e4_4.exp_l14e4_4.exp_l14e5_4.exp_l15e1_4.exp_l15e2_4.exp_l16e7_4.exp_l16e6_4.exp_l17e2_4.exp_l17e5_4.exp_l18e0_4.exp_l18e2_4.exp_l19e0_4.exp_l19e4_4.exp_l20e6_4.exp_l20e7_4.exp_l21e6_4.exp_l21e3_4.exp_l22e0_4.exp_l22e3_4.exp_l23e1_4.exp_l23e0_4.exp_l24e2_4.exp_l24e1_4.exp_l25e1_4.exp_l25e5_4.exp_l26e3_4.exp_l26e2_4.exp_l27e5_4.exp_l27e0_4.exp_l28e1_4.exp_l28e3_4.exp_l29e2_4.exp_l29e6_4.exp_l30e7_4.exp_l30e4_4.exp_l31e7_4.exp_l31e3_4 \
      --bits_name main_2.attn_4.each_layer_top2

# Attention 4 bits + Layer selected by cosine similarity predictors 4 bits (128 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.layer_29_4.layer_11_4.layer_7_4.layer_9_4.layer_14_4.layer_15_4.layer_28_4.layer_6_4.layer_16_4.layer_5_4 \
      --bits_name main_2.attn_4.top16_cos_sim_layer_avg_3bits

DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e0_4.exp_l0e1_4.exp_l0e2_4.exp_l0e3_4.exp_l0e4_4.exp_l0e5_4.exp_l0e6_4.exp_l0e7_4.exp_l1e0_4.exp_l1e1_4.exp_l1e2_4.exp_l1e3_4.exp_l1e4_4.exp_l1e5_4.exp_l1e6_4.exp_l1e7_4.exp_l19e0_4.exp_l19e1_4.exp_l19e2_4.exp_l19e3_4.exp_l19e4_4.exp_l19e5_4.exp_l19e6_4.exp_l19e7_4.exp_l18e0_4.exp_l18e1_4.exp_l18e2_4.exp_l18e3_4.exp_l18e4_4.exp_l18e5_4.exp_l18e6_4.exp_l18e7_4.exp_l2e0_4.exp_l2e1_4.exp_l2e2_4.exp_l2e3_4.exp_l2e4_4.exp_l2e5_4.exp_l2e6_4.exp_l2e7_4.exp_l20e0_4.exp_l20e1_4.exp_l20e2_4.exp_l20e3_4.exp_l20e4_4.exp_l20e5_4.exp_l20e6_4.exp_l20e7_4.exp_l3e0_4.exp_l3e1_4.exp_l3e2_4.exp_l3e3_4.exp_l3e4_4.exp_l3e5_4.exp_l3e6_4.exp_l3e7_4.exp_l4e0_4.exp_l4e1_4.exp_l4e2_4.exp_l4e3_4.exp_l4e4_4.exp_l4e5_4.exp_l4e6_4.exp_l4e7_4.exp_l21e0_4.exp_l21e1_4.exp_l21e2_4.exp_l21e3_4.exp_l21e4_4.exp_l21e5_4.exp_l21e6_4.exp_l21e7_4.exp_l22e0_4.exp_l22e1_4.exp_l22e2_4.exp_l22e3_4.exp_l22e4_4.exp_l22e5_4.exp_l22e6_4.exp_l22e7_4.exp_l23e0_4.exp_l23e1_4.exp_l23e2_4.exp_l23e3_4.exp_l23e4_4.exp_l23e5_4.exp_l23e6_4.exp_l23e7_4.exp_l17e0_4.exp_l17e1_4.exp_l17e2_4.exp_l17e3_4.exp_l17e4_4.exp_l17e5_4.exp_l17e6_4.exp_l17e7_4.exp_l26e0_4.exp_l26e1_4.exp_l26e2_4.exp_l26e3_4.exp_l26e4_4.exp_l26e5_4.exp_l26e6_4.exp_l26e7_4.exp_l24e0_4.exp_l24e1_4.exp_l24e2_4.exp_l24e3_4.exp_l24e4_4.exp_l24e5_4.exp_l24e6_4.exp_l24e7_4.exp_l25e0_4.exp_l25e1_4.exp_l25e2_4.exp_l25e3_4.exp_l25e4_4.exp_l25e5_4.exp_l25e6_4.exp_l25e7_4.exp_l27e0_4.exp_l27e1_4.exp_l27e2_4.exp_l27e3_4.exp_l27e4_4.exp_l27e5_4.exp_l27e6_4.exp_l27e7_4 \
      --bits_name main_2.attn_4.bottom16_cos_sim_layer_avg_3bits

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l31e0_4.exp_l31e1_4.exp_l31e2_4.exp_l31e3_4.exp_l31e4_4.exp_l31e5_4.exp_l31e6_4.exp_l31e7_4.exp_l0e0_4.exp_l0e1_4.exp_l0e2_4.exp_l0e3_4.exp_l0e4_4.exp_l0e5_4.exp_l0e6_4.exp_l0e7_4.exp_l3e0_4.exp_l3e1_4.exp_l3e2_4.exp_l3e3_4.exp_l3e4_4.exp_l3e5_4.exp_l3e6_4.exp_l3e7_4.exp_l30e0_4.exp_l30e1_4.exp_l30e2_4.exp_l30e3_4.exp_l30e4_4.exp_l30e5_4.exp_l30e6_4.exp_l30e7_4.exp_l1e0_4.exp_l1e1_4.exp_l1e2_4.exp_l1e3_4.exp_l1e4_4.exp_l1e5_4.exp_l1e6_4.exp_l1e7_4.exp_l2e0_4.exp_l2e1_4.exp_l2e2_4.exp_l2e3_4.exp_l2e4_4.exp_l2e5_4.exp_l2e6_4.exp_l2e7_4.exp_l4e0_4.exp_l4e1_4.exp_l4e2_4.exp_l4e3_4.exp_l4e4_4.exp_l4e5_4.exp_l4e6_4.exp_l4e7_4.exp_l5e0_4.exp_l5e1_4.exp_l5e2_4.exp_l5e3_4.exp_l5e4_4.exp_l5e5_4.exp_l5e6_4.exp_l5e7_4.exp_l6e0_4.exp_l6e1_4.exp_l6e2_4.exp_l6e3_4.exp_l6e4_4.exp_l6e5_4.exp_l6e6_4.exp_l6e7_4.exp_l29e0_4.exp_l29e1_4.exp_l29e2_4.exp_l29e3_4.exp_l29e4_4.exp_l29e5_4.exp_l29e6_4.exp_l29e7_4.exp_l8e0_4.exp_l8e1_4.exp_l8e2_4.exp_l8e3_4.exp_l8e4_4.exp_l8e5_4.exp_l8e6_4.exp_l8e7_4.exp_l18e0_4.exp_l18e1_4.exp_l18e2_4.exp_l18e3_4.exp_l18e4_4.exp_l18e5_4.exp_l18e6_4.exp_l18e7_4.exp_l7e0_4.exp_l7e1_4.exp_l7e2_4.exp_l7e3_4.exp_l7e4_4.exp_l7e5_4.exp_l7e6_4.exp_l7e7_4.exp_l17e0_4.exp_l17e1_4.exp_l17e2_4.exp_l17e3_4.exp_l17e4_4.exp_l17e5_4.exp_l17e6_4.exp_l17e7_4.exp_l14e0_4.exp_l14e1_4.exp_l14e2_4.exp_l14e3_4.exp_l14e4_4.exp_l14e5_4.exp_l14e6_4.exp_l14e7_4.exp_l15e0_4.exp_l15e1_4.exp_l15e2_4.exp_l15e3_4.exp_l15e4_4.exp_l15e5_4.exp_l15e6_4.exp_l15e7_4 \
      --bits_name main_2.attn_4.top16_residual_cos_sim_layer_avg_3bits

# 192 experts
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l31e0_4.exp_l31e1_4.exp_l31e2_4.exp_l31e3_4.exp_l31e4_4.exp_l31e5_4.exp_l31e6_4.exp_l31e7_4.exp_l30e0_4.exp_l30e1_4.exp_l30e2_4.exp_l30e3_4.exp_l30e4_4.exp_l30e5_4.exp_l30e6_4.exp_l30e7_4.exp_l12e0_4.exp_l12e1_4.exp_l12e2_4.exp_l12e3_4.exp_l12e4_4.exp_l12e5_4.exp_l12e6_4.exp_l12e7_4.exp_l10e0_4.exp_l10e1_4.exp_l10e2_4.exp_l10e3_4.exp_l10e4_4.exp_l10e5_4.exp_l10e6_4.exp_l10e7_4.exp_l13e0_4.exp_l13e1_4.exp_l13e2_4.exp_l13e3_4.exp_l13e4_4.exp_l13e5_4.exp_l13e6_4.exp_l13e7_4.exp_l8e0_4.exp_l8e1_4.exp_l8e2_4.exp_l8e3_4.exp_l8e4_4.exp_l8e5_4.exp_l8e6_4.exp_l8e7_4.exp_l29e0_4.exp_l29e1_4.exp_l29e2_4.exp_l29e3_4.exp_l29e4_4.exp_l29e5_4.exp_l29e6_4.exp_l29e7_4.exp_l11e0_4.exp_l11e1_4.exp_l11e2_4.exp_l11e3_4.exp_l11e4_4.exp_l11e5_4.exp_l11e6_4.exp_l11e7_4.exp_l7e0_4.exp_l7e1_4.exp_l7e2_4.exp_l7e3_4.exp_l7e4_4.exp_l7e5_4.exp_l7e6_4.exp_l7e7_4.exp_l9e0_4.exp_l9e1_4.exp_l9e2_4.exp_l9e3_4.exp_l9e4_4.exp_l9e5_4.exp_l9e6_4.exp_l9e7_4.exp_l14e0_4.exp_l14e1_4.exp_l14e2_4.exp_l14e3_4.exp_l14e4_4.exp_l14e5_4.exp_l14e6_4.exp_l14e7_4.exp_l15e0_4.exp_l15e1_4.exp_l15e2_4.exp_l15e3_4.exp_l15e4_4.exp_l15e5_4.exp_l15e6_4.exp_l15e7_4.exp_l28e0_4.exp_l28e1_4.exp_l28e2_4.exp_l28e3_4.exp_l28e4_4.exp_l28e5_4.exp_l28e6_4.exp_l28e7_4.exp_l6e0_4.exp_l6e1_4.exp_l6e2_4.exp_l6e3_4.exp_l6e4_4.exp_l6e5_4.exp_l6e6_4.exp_l6e7_4.exp_l16e0_4.exp_l16e1_4.exp_l16e2_4.exp_l16e3_4.exp_l16e4_4.exp_l16e5_4.exp_l16e6_4.exp_l16e7_4.exp_l5e0_4.exp_l5e1_4.exp_l5e2_4.exp_l5e3_4.exp_l5e4_4.exp_l5e5_4.exp_l5e6_4.exp_l5e7_4.exp_l27e0_4.exp_l27e1_4.exp_l27e2_4.exp_l27e3_4.exp_l27e4_4.exp_l27e5_4.exp_l27e6_4.exp_l27e7_4.exp_l25e0_4.exp_l25e1_4.exp_l25e2_4.exp_l25e3_4.exp_l25e4_4.exp_l25e5_4.exp_l25e6_4.exp_l25e7_4.exp_l24e0_4.exp_l24e1_4.exp_l24e2_4.exp_l24e3_4.exp_l24e4_4.exp_l24e5_4.exp_l24e6_4.exp_l24e7_4.exp_l26e0_4.exp_l26e1_4.exp_l26e2_4.exp_l26e3_4.exp_l26e4_4.exp_l26e5_4.exp_l26e6_4.exp_l26e7_4.exp_l17e0_4.exp_l17e1_4.exp_l17e2_4.exp_l17e3_4.exp_l17e4_4.exp_l17e5_4.exp_l17e6_4.exp_l17e7_4.exp_l23e0_4.exp_l23e1_4.exp_l23e2_4.exp_l23e3_4.exp_l23e4_4.exp_l23e5_4.exp_l23e6_4.exp_l23e7_4.exp_l22e0_4.exp_l22e1_4.exp_l22e2_4.exp_l22e3_4.exp_l22e4_4.exp_l22e5_4.exp_l22e6_4.exp_l22e7_4.exp_l21e0_4.exp_l21e1_4.exp_l21e2_4.exp_l21e3_4.exp_l21e4_4.exp_l21e5_4.exp_l21e6_4.exp_l21e7_4 \
      --bits_name main_2.attn_4.top24_cos_sim_layer

# 64 experts
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l31e0_4.exp_l31e1_4.exp_l31e2_4.exp_l31e3_4.exp_l31e4_4.exp_l31e5_4.exp_l31e6_4.exp_l31e7_4.exp_l30e0_4.exp_l30e1_4.exp_l30e2_4.exp_l30e3_4.exp_l30e4_4.exp_l30e5_4.exp_l30e6_4.exp_l30e7_4.exp_l12e0_4.exp_l12e1_4.exp_l12e2_4.exp_l12e3_4.exp_l12e4_4.exp_l12e5_4.exp_l12e6_4.exp_l12e7_4.exp_l10e0_4.exp_l10e1_4.exp_l10e2_4.exp_l10e3_4.exp_l10e4_4.exp_l10e5_4.exp_l10e6_4.exp_l10e7_4.exp_l13e0_4.exp_l13e1_4.exp_l13e2_4.exp_l13e3_4.exp_l13e4_4.exp_l13e5_4.exp_l13e6_4.exp_l13e7_4.exp_l8e0_4.exp_l8e1_4.exp_l8e2_4.exp_l8e3_4.exp_l8e4_4.exp_l8e5_4.exp_l8e6_4.exp_l8e7_4.exp_l29e0_4.exp_l29e1_4.exp_l29e2_4.exp_l29e3_4.exp_l29e4_4.exp_l29e5_4.exp_l29e6_4.exp_l29e7_4.exp_l11e0_4.exp_l11e1_4.exp_l11e2_4.exp_l11e3_4.exp_l11e4_4.exp_l11e5_4.exp_l11e6_4.exp_l11e7_4 \
      --bits_name main_2.attn_4.top8_cos_sim_layer

# 224 experts
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l31e0_4.exp_l31e1_4.exp_l31e2_4.exp_l31e3_4.exp_l31e4_4.exp_l31e5_4.exp_l31e6_4.exp_l31e7_4.exp_l30e0_4.exp_l30e1_4.exp_l30e2_4.exp_l30e3_4.exp_l30e4_4.exp_l30e5_4.exp_l30e6_4.exp_l30e7_4.exp_l12e0_4.exp_l12e1_4.exp_l12e2_4.exp_l12e3_4.exp_l12e4_4.exp_l12e5_4.exp_l12e6_4.exp_l12e7_4.exp_l10e0_4.exp_l10e1_4.exp_l10e2_4.exp_l10e3_4.exp_l10e4_4.exp_l10e5_4.exp_l10e6_4.exp_l10e7_4.exp_l13e0_4.exp_l13e1_4.exp_l13e2_4.exp_l13e3_4.exp_l13e4_4.exp_l13e5_4.exp_l13e6_4.exp_l13e7_4.exp_l8e0_4.exp_l8e1_4.exp_l8e2_4.exp_l8e3_4.exp_l8e4_4.exp_l8e5_4.exp_l8e6_4.exp_l8e7_4.exp_l29e0_4.exp_l29e1_4.exp_l29e2_4.exp_l29e3_4.exp_l29e4_4.exp_l29e5_4.exp_l29e6_4.exp_l29e7_4.exp_l11e0_4.exp_l11e1_4.exp_l11e2_4.exp_l11e3_4.exp_l11e4_4.exp_l11e5_4.exp_l11e6_4.exp_l11e7_4.exp_l7e0_4.exp_l7e1_4.exp_l7e2_4.exp_l7e3_4.exp_l7e4_4.exp_l7e5_4.exp_l7e6_4.exp_l7e7_4.exp_l9e0_4.exp_l9e1_4.exp_l9e2_4.exp_l9e3_4.exp_l9e4_4.exp_l9e5_4.exp_l9e6_4.exp_l9e7_4.exp_l14e0_4.exp_l14e1_4.exp_l14e2_4.exp_l14e3_4.exp_l14e4_4.exp_l14e5_4.exp_l14e6_4.exp_l14e7_4.exp_l15e0_4.exp_l15e1_4.exp_l15e2_4.exp_l15e3_4.exp_l15e4_4.exp_l15e5_4.exp_l15e6_4.exp_l15e7_4.exp_l28e0_4.exp_l28e1_4.exp_l28e2_4.exp_l28e3_4.exp_l28e4_4.exp_l28e5_4.exp_l28e6_4.exp_l28e7_4.exp_l6e0_4.exp_l6e1_4.exp_l6e2_4.exp_l6e3_4.exp_l6e4_4.exp_l6e5_4.exp_l6e6_4.exp_l6e7_4.exp_l16e0_4.exp_l16e1_4.exp_l16e2_4.exp_l16e3_4.exp_l16e4_4.exp_l16e5_4.exp_l16e6_4.exp_l16e7_4.exp_l5e0_4.exp_l5e1_4.exp_l5e2_4.exp_l5e3_4.exp_l5e4_4.exp_l5e5_4.exp_l5e6_4.exp_l5e7_4.exp_l27e0_4.exp_l27e1_4.exp_l27e2_4.exp_l27e3_4.exp_l27e4_4.exp_l27e5_4.exp_l27e6_4.exp_l27e7_4.exp_l25e0_4.exp_l25e1_4.exp_l25e2_4.exp_l25e3_4.exp_l25e4_4.exp_l25e5_4.exp_l25e6_4.exp_l25e7_4.exp_l24e0_4.exp_l24e1_4.exp_l24e2_4.exp_l24e3_4.exp_l24e4_4.exp_l24e5_4.exp_l24e6_4.exp_l24e7_4.exp_l26e0_4.exp_l26e1_4.exp_l26e2_4.exp_l26e3_4.exp_l26e4_4.exp_l26e5_4.exp_l26e6_4.exp_l26e7_4.exp_l17e0_4.exp_l17e1_4.exp_l17e2_4.exp_l17e3_4.exp_l17e4_4.exp_l17e5_4.exp_l17e6_4.exp_l17e7_4.exp_l23e0_4.exp_l23e1_4.exp_l23e2_4.exp_l23e3_4.exp_l23e4_4.exp_l23e5_4.exp_l23e6_4.exp_l23e7_4.exp_l22e0_4.exp_l22e1_4.exp_l22e2_4.exp_l22e3_4.exp_l22e4_4.exp_l22e5_4.exp_l22e6_4.exp_l22e7_4.exp_l21e0_4.exp_l21e1_4.exp_l21e2_4.exp_l21e3_4.exp_l21e4_4.exp_l21e5_4.exp_l21e6_4.exp_l21e7_4.exp_l4e0_4.exp_l4e1_4.exp_l4e2_4.exp_l4e3_4.exp_l4e4_4.exp_l4e5_4.exp_l4e6_4.exp_l4e7_4.exp_l3e0_4.exp_l3e1_4.exp_l3e2_4.exp_l3e3_4.exp_l3e4_4.exp_l3e5_4.exp_l3e6_4.exp_l3e7_4.exp_l20e0_4.exp_l20e1_4.exp_l20e2_4.exp_l20e3_4.exp_l20e4_4.exp_l20e5_4.exp_l20e6_4.exp_l20e7_4.exp_l2e0_4.exp_l2e1_4.exp_l2e2_4.exp_l2e3_4.exp_l2e4_4.exp_l2e5_4.exp_l2e6_4.exp_l2e7_4 \
      --bits_name main_2.attn_4.top28_cos_sim_layer

# 128 + 8 experts (top16 cos sim + layer 1)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.layer_29_4.layer_11_4.layer_7_4.layer_9_4.layer_14_4.layer_15_4.layer_28_4.layer_6_4.layer_16_4.layer_5_4.layer_1_4 \
      --bits_name main_2.attn_4.top16_cos_sim_layer_and_layer1

# cos-sim top 12 layers cos sim (96 experts)
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.layer_29_4.layer_11_4.layer_7_4.layer_9_4.layer_14_4.layer_15_4 \
      --bits_name main_2.attn_4.top12_cos_sim_layer

# cos-sim top 6 layers + all w2
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.keyword__w2__4 \
      --bits_name main_2.attn_4.top6_cos_sim_and_all_w2

# cos-sim top 6 layers + all w1
DEBUG=0 CUDA_VISIBLE_DEVICES=7 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.keyword__w1__4 \
      --bits_name main_2.attn_4.top6_cos_sim_and_all_w1

# cos-sim top 6 layers + all w3
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.keyword__w3__4 \
      --bits_name main_2.attn_4.top6_cos_sim_and_all_w3

# cos-sim top 6 layers + all w2 + layer 0
DEBUG=0 CUDA_VISIBLE_DEVICES=1 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.keyword__w2__4 \
      --bits_name main_2.attn_4.top6_cos_sim_and_layer0_and_all_w2

# layerset 1: selected manually from the channel-max plot + all w2
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_13_4.layer_16_4.layer_18_4.layer_20_4.layer_23_4.layer_26_4.keyword__w2__4 \
      --bits_name main_2.attn_4.layerset_1_and_all_w2

# layerset 2  + all w2
DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_8_4.layer_10_4.layer_12_4.layer_13_4.layer_16_4.layer_18_4.layer_20_4.layer_23_4.layer_26_4.layer_30_4.layer_31_4.keyword__w2__4 \
      --bits_name main_2.attn_4.layerset_2_and_all_w2

# layerset 3  + all w2
DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_8_4.layer_10_4.layer_11_4.layer_12_4.layer_13_4.layer_16_4.layer_18_4.layer_20_4.layer_23_4.layer_26_4.layer_29_4.layer_30_4.layer_31_4.keyword__w2__4 \
      --bits_name main_2.attn_4.layerset_3_and_all_w2

# layerset 4  + all w2
DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_7_4.layer_8_4.layer_9_4.layer_10_4.layer_11_4.layer_12_4.layer_13_4.layer_16_4.layer_18_4.layer_20_4.layer_23_4.layer_26_4.layer_29_4.layer_30_4.layer_31_4.keyword__w2__4 \
      --bits_name main_2.attn_4.layerset_4_and_all_w2

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_7_4.layer_8_4.layer_9_4.layer_10_4.layer_12_4.layer_13_4.layer_16_4.layer_18_4.layer_20_4.layer_23_4.layer_26_4.layer_30_4.layer_31_4.keyword__w2__4 \
      --bits_name main_2.attn_4.layerset_5_and_all_w2

# all w1/w2/w3
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.keyword__w1__4 \
      --bits_name main_2.attn_4.all_w1

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.keyword__w2__4 \
      --bits_name main_2.attn_4.all_w2

DEBUG=0 CUDA_VISIBLE_DEVICES=2 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.keyword__w3__4 \
      --bits_name main_2.attn_4.all_w3

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_4.attn_3 \
      --bits_name main_4.attn_3

# attn 4 + frequency top-2 per layer (2.54 bits)
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l0e2_4.exp_l1e4_4.exp_l1e3_4.exp_l2e0_4.exp_l2e1_4.exp_l3e2_4.exp_l3e5_4.exp_l4e3_4.exp_l4e2_4.exp_l5e1_4.exp_l5e2_4.exp_l6e5_4.exp_l6e0_4.exp_l7e5_4.exp_l7e6_4.exp_l8e5_4.exp_l8e0_4.exp_l9e0_4.exp_l9e3_4.exp_l10e1_4.exp_l10e2_4.exp_l11e1_4.exp_l11e4_4.exp_l12e1_4.exp_l12e2_4.exp_l13e7_4.exp_l13e4_4.exp_l14e4_4.exp_l14e5_4.exp_l15e1_4.exp_l15e2_4.exp_l16e7_4.exp_l16e6_4.exp_l17e2_4.exp_l17e5_4.exp_l18e0_4.exp_l18e2_4.exp_l19e0_4.exp_l19e4_4.exp_l20e6_4.exp_l20e7_4.exp_l21e6_4.exp_l21e3_4.exp_l22e0_4.exp_l22e3_4.exp_l23e1_4.exp_l23e0_4.exp_l24e2_4.exp_l24e1_4.exp_l25e1_4.exp_l25e5_4.exp_l26e3_4.exp_l26e2_4.exp_l27e5_4.exp_l27e0_4.exp_l28e1_4.exp_l28e3_4.exp_l29e2_4.exp_l29e6_4.exp_l30e7_4.exp_l30e4_4.exp_l31e7_4.exp_l31e3_4 \
      --bits_name main_2.attn_4.frequency_top2_per_layer

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e6_4.exp_l0e3_4.exp_l1e4_4.exp_l1e1_4.exp_l2e7_4.exp_l2e3_4.exp_l3e4_4.exp_l3e1_4.exp_l4e7_4.exp_l4e3_4.exp_l5e4_4.exp_l5e5_4.exp_l6e1_4.exp_l6e6_4.exp_l7e5_4.exp_l7e2_4.exp_l8e7_4.exp_l8e6_4.exp_l9e0_4.exp_l9e6_4.exp_l10e1_4.exp_l10e5_4.exp_l11e3_4.exp_l11e7_4.exp_l12e6_4.exp_l12e3_4.exp_l13e3_4.exp_l13e2_4.exp_l14e5_4.exp_l14e7_4.exp_l15e1_4.exp_l15e0_4.exp_l16e7_4.exp_l16e2_4.exp_l17e4_4.exp_l17e6_4.exp_l18e1_4.exp_l18e3_4.exp_l19e4_4.exp_l19e7_4.exp_l20e7_4.exp_l20e4_4.exp_l21e2_4.exp_l21e7_4.exp_l22e4_4.exp_l22e3_4.exp_l23e0_4.exp_l23e1_4.exp_l24e6_4.exp_l24e0_4.exp_l25e5_4.exp_l25e1_4.exp_l26e5_4.exp_l26e0_4.exp_l27e5_4.exp_l27e2_4.exp_l28e1_4.exp_l28e7_4.exp_l29e0_4.exp_l29e7_4.exp_l30e6_4.exp_l30e1_4.exp_l31e4_4.exp_l31e3_4 \
      --bits_name main_2.attn_4.random2_per_layer_seed42

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e4_4.exp_l0e5_4.exp_l1e3_4.exp_l1e2_4.exp_l2e7_4.exp_l2e1_4.exp_l3e1_4.exp_l3e3_4.exp_l4e5_4.exp_l4e0_4.exp_l5e4_4.exp_l5e0_4.exp_l6e2_4.exp_l6e3_4.exp_l7e7_4.exp_l7e0_4.exp_l8e3_4.exp_l8e6_4.exp_l9e6_4.exp_l9e3_4.exp_l10e2_4.exp_l10e7_4.exp_l11e6_4.exp_l11e7_4.exp_l12e7_4.exp_l12e5_4.exp_l13e5_4.exp_l13e7_4.exp_l14e5_4.exp_l14e3_4.exp_l15e2_4.exp_l15e3_4.exp_l16e7_4.exp_l16e3_4.exp_l17e3_4.exp_l17e0_4.exp_l18e6_4.exp_l18e3_4.exp_l19e3_4.exp_l19e1_4.exp_l20e4_4.exp_l20e3_4.exp_l21e7_4.exp_l21e1_4.exp_l22e3_4.exp_l22e0_4.exp_l23e0_4.exp_l23e4_4.exp_l24e4_4.exp_l24e3_4.exp_l25e4_4.exp_l25e6_4.exp_l26e5_4.exp_l26e6_4.exp_l27e7_4.exp_l27e0_4.exp_l28e1_4.exp_l28e5_4.exp_l29e6_4.exp_l29e0_4.exp_l30e4_4.exp_l30e0_4.exp_l31e5_4.exp_l31e7_4 \
      --bits_name main_2.attn_4.random2_per_layer_seed43

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e4_4.exp_l0e7_4.exp_l1e4_4.exp_l1e0_4.exp_l2e6_4.exp_l2e7_4.exp_l3e5_4.exp_l3e2_4.exp_l4e0_4.exp_l4e5_4.exp_l5e2_4.exp_l5e5_4.exp_l6e5_4.exp_l6e0_4.exp_l7e3_4.exp_l7e7_4.exp_l8e5_4.exp_l8e3_4.exp_l9e2_4.exp_l9e1_4.exp_l10e2_4.exp_l10e6_4.exp_l11e4_4.exp_l11e2_4.exp_l12e3_4.exp_l12e1_4.exp_l13e4_4.exp_l13e6_4.exp_l14e0_4.exp_l14e6_4.exp_l15e0_4.exp_l15e4_4.exp_l16e7_4.exp_l16e1_4.exp_l17e6_4.exp_l17e2_4.exp_l18e0_4.exp_l18e2_4.exp_l19e0_4.exp_l19e5_4.exp_l20e6_4.exp_l20e7_4.exp_l21e3_4.exp_l21e4_4.exp_l22e2_4.exp_l22e6_4.exp_l23e0_4.exp_l23e1_4.exp_l24e4_4.exp_l24e1_4.exp_l25e1_4.exp_l25e0_4.exp_l26e1_4.exp_l26e6_4.exp_l27e5_4.exp_l27e6_4.exp_l28e2_4.exp_l28e4_4.exp_l29e2_4.exp_l29e5_4.exp_l30e7_4.exp_l30e0_4.exp_l31e1_4.exp_l31e2_4 \
      --bits_name main_2.attn_4.random2_per_layer_seed44


DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e5_4.exp_l0e2_4.exp_l0e7_4.exp_l0e0_4.exp_l1e4_4.exp_l1e3_4.exp_l1e0_4.exp_l1e5_4.exp_l2e0_4.exp_l2e1_4.exp_l2e6_4.exp_l2e5_4.exp_l3e2_4.exp_l3e5_4.exp_l3e6_4.exp_l3e3_4.exp_l4e3_4.exp_l4e2_4.exp_l4e7_4.exp_l4e1_4.exp_l5e1_4.exp_l5e2_4.exp_l5e5_4.exp_l5e0_4.exp_l6e5_4.exp_l6e0_4.exp_l6e7_4.exp_l6e6_4.exp_l7e5_4.exp_l7e6_4.exp_l7e4_4.exp_l7e0_4.exp_l8e5_4.exp_l8e0_4.exp_l8e6_4.exp_l8e4_4.exp_l9e0_4.exp_l9e3_4.exp_l9e6_4.exp_l9e5_4.exp_l10e1_4.exp_l10e2_4.exp_l10e0_4.exp_l10e3_4.exp_l11e1_4.exp_l11e4_4.exp_l11e2_4.exp_l11e0_4.exp_l12e1_4.exp_l12e2_4.exp_l12e0_4.exp_l12e3_4.exp_l13e7_4.exp_l13e4_4.exp_l13e6_4.exp_l13e5_4.exp_l14e4_4.exp_l14e5_4.exp_l14e3_4.exp_l14e1_4.exp_l15e1_4.exp_l15e2_4.exp_l15e0_4.exp_l15e7_4.exp_l16e7_4.exp_l16e6_4.exp_l16e5_4.exp_l16e4_4.exp_l17e2_4.exp_l17e5_4.exp_l17e7_4.exp_l17e1_4.exp_l18e0_4.exp_l18e2_4.exp_l18e1_4.exp_l18e4_4.exp_l19e0_4.exp_l19e4_4.exp_l19e5_4.exp_l19e2_4.exp_l20e6_4.exp_l20e7_4.exp_l20e2_4.exp_l20e5_4.exp_l21e6_4.exp_l21e3_4.exp_l21e0_4.exp_l21e7_4.exp_l22e0_4.exp_l22e3_4.exp_l22e5_4.exp_l22e7_4.exp_l23e1_4.exp_l23e0_4.exp_l23e6_4.exp_l23e7_4.exp_l24e2_4.exp_l24e1_4.exp_l24e5_4.exp_l24e4_4.exp_l25e1_4.exp_l25e5_4.exp_l25e4_4.exp_l25e3_4.exp_l26e3_4.exp_l26e2_4.exp_l26e6_4.exp_l26e4_4.exp_l27e5_4.exp_l27e0_4.exp_l27e2_4.exp_l27e4_4.exp_l28e1_4.exp_l28e3_4.exp_l28e4_4.exp_l28e0_4.exp_l29e2_4.exp_l29e6_4.exp_l29e7_4.exp_l29e1_4.exp_l30e7_4.exp_l30e4_4.exp_l30e2_4.exp_l30e1_4.exp_l31e7_4.exp_l31e3_4.exp_l31e2_4.exp_l31e5_4 \
      --bits_name main_2.attn_4.frequency_top4_per_layer

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e6_4.exp_l0e3_4.exp_l0e0_4.exp_l0e7_4.exp_l1e4_4.exp_l1e1_4.exp_l1e7_4.exp_l1e6_4.exp_l2e7_4.exp_l2e3_4.exp_l2e0_4.exp_l2e1_4.exp_l3e4_4.exp_l3e1_4.exp_l3e7_4.exp_l3e2_4.exp_l4e7_4.exp_l4e3_4.exp_l4e2_4.exp_l4e1_4.exp_l5e4_4.exp_l5e5_4.exp_l5e0_4.exp_l5e1_4.exp_l6e1_4.exp_l6e6_4.exp_l6e5_4.exp_l6e0_4.exp_l7e5_4.exp_l7e2_4.exp_l7e0_4.exp_l7e3_4.exp_l8e7_4.exp_l8e6_4.exp_l8e4_4.exp_l8e1_4.exp_l9e0_4.exp_l9e6_4.exp_l9e7_4.exp_l9e1_4.exp_l10e1_4.exp_l10e5_4.exp_l10e6_4.exp_l10e2_4.exp_l11e3_4.exp_l11e7_4.exp_l11e1_4.exp_l11e2_4.exp_l12e6_4.exp_l12e3_4.exp_l12e1_4.exp_l12e0_4.exp_l13e3_4.exp_l13e2_4.exp_l13e5_4.exp_l13e4_4.exp_l14e5_4.exp_l14e7_4.exp_l14e6_4.exp_l14e2_4.exp_l15e1_4.exp_l15e0_4.exp_l15e3_4.exp_l15e2_4.exp_l16e7_4.exp_l16e2_4.exp_l16e4_4.exp_l16e0_4.exp_l17e4_4.exp_l17e6_4.exp_l17e1_4.exp_l17e5_4.exp_l18e1_4.exp_l18e3_4.exp_l18e5_4.exp_l18e4_4.exp_l19e4_4.exp_l19e7_4.exp_l19e6_4.exp_l19e0_4.exp_l20e7_4.exp_l20e4_4.exp_l20e0_4.exp_l20e2_4.exp_l21e2_4.exp_l21e7_4.exp_l21e3_4.exp_l21e5_4.exp_l22e4_4.exp_l22e3_4.exp_l22e2_4.exp_l22e6_4.exp_l23e0_4.exp_l23e1_4.exp_l23e5_4.exp_l23e6_4.exp_l24e6_4.exp_l24e0_4.exp_l24e3_4.exp_l24e7_4.exp_l25e5_4.exp_l25e1_4.exp_l25e2_4.exp_l25e6_4.exp_l26e5_4.exp_l26e0_4.exp_l26e6_4.exp_l26e4_4.exp_l27e5_4.exp_l27e2_4.exp_l27e4_4.exp_l27e0_4.exp_l28e1_4.exp_l28e7_4.exp_l28e6_4.exp_l28e0_4.exp_l29e0_4.exp_l29e7_4.exp_l29e2_4.exp_l29e3_4.exp_l30e6_4.exp_l30e1_4.exp_l30e2_4.exp_l30e5_4.exp_l31e4_4.exp_l31e3_4.exp_l31e0_4.exp_l31e6_4 \
      --bits_name main_2.attn_4.random4_per_layer_seed42

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e4_4.exp_l0e5_4.exp_l0e7_4.exp_l0e1_4.exp_l1e3_4.exp_l1e2_4.exp_l1e0_4.exp_l1e4_4.exp_l2e7_4.exp_l2e1_4.exp_l2e0_4.exp_l2e2_4.exp_l3e1_4.exp_l3e3_4.exp_l3e0_4.exp_l3e5_4.exp_l4e5_4.exp_l4e0_4.exp_l4e4_4.exp_l4e3_4.exp_l5e4_4.exp_l5e0_4.exp_l5e6_4.exp_l5e7_4.exp_l6e2_4.exp_l6e3_4.exp_l6e4_4.exp_l6e1_4.exp_l7e7_4.exp_l7e0_4.exp_l7e2_4.exp_l7e6_4.exp_l8e3_4.exp_l8e6_4.exp_l8e5_4.exp_l8e4_4.exp_l9e6_4.exp_l9e3_4.exp_l9e0_4.exp_l9e7_4.exp_l10e2_4.exp_l10e7_4.exp_l10e4_4.exp_l10e5_4.exp_l11e6_4.exp_l11e7_4.exp_l11e4_4.exp_l11e3_4.exp_l12e7_4.exp_l12e5_4.exp_l12e0_4.exp_l12e6_4.exp_l13e5_4.exp_l13e7_4.exp_l13e1_4.exp_l13e0_4.exp_l14e5_4.exp_l14e3_4.exp_l14e6_4.exp_l14e7_4.exp_l15e2_4.exp_l15e3_4.exp_l15e0_4.exp_l15e6_4.exp_l16e7_4.exp_l16e3_4.exp_l16e5_4.exp_l16e6_4.exp_l17e3_4.exp_l17e0_4.exp_l17e5_4.exp_l17e6_4.exp_l18e6_4.exp_l18e3_4.exp_l18e0_4.exp_l18e4_4.exp_l19e3_4.exp_l19e1_4.exp_l19e5_4.exp_l19e0_4.exp_l20e4_4.exp_l20e3_4.exp_l20e6_4.exp_l20e7_4.exp_l21e7_4.exp_l21e1_4.exp_l21e0_4.exp_l21e6_4.exp_l22e3_4.exp_l22e0_4.exp_l22e7_4.exp_l22e1_4.exp_l23e0_4.exp_l23e4_4.exp_l23e2_4.exp_l23e5_4.exp_l24e4_4.exp_l24e3_4.exp_l24e7_4.exp_l24e2_4.exp_l25e4_4.exp_l25e6_4.exp_l25e2_4.exp_l25e7_4.exp_l26e5_4.exp_l26e6_4.exp_l26e4_4.exp_l26e1_4.exp_l27e7_4.exp_l27e0_4.exp_l27e1_4.exp_l27e3_4.exp_l28e1_4.exp_l28e5_4.exp_l28e3_4.exp_l28e7_4.exp_l29e6_4.exp_l29e0_4.exp_l29e4_4.exp_l29e7_4.exp_l30e4_4.exp_l30e0_4.exp_l30e5_4.exp_l30e6_4.exp_l31e5_4.exp_l31e7_4.exp_l31e4_4.exp_l31e2_4 \
      --bits_name main_2.attn_4.random4_per_layer_seed43

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.exp_l0e4_4.exp_l0e7_4.exp_l0e3_4.exp_l0e2_4.exp_l1e4_4.exp_l1e0_4.exp_l1e7_4.exp_l1e5_4.exp_l2e6_4.exp_l2e7_4.exp_l2e1_4.exp_l2e4_4.exp_l3e5_4.exp_l3e2_4.exp_l3e1_4.exp_l3e4_4.exp_l4e0_4.exp_l4e5_4.exp_l4e6_4.exp_l4e2_4.exp_l5e2_4.exp_l5e5_4.exp_l5e1_4.exp_l5e4_4.exp_l6e5_4.exp_l6e0_4.exp_l6e4_4.exp_l6e6_4.exp_l7e3_4.exp_l7e7_4.exp_l7e6_4.exp_l7e5_4.exp_l8e5_4.exp_l8e3_4.exp_l8e7_4.exp_l8e1_4.exp_l9e2_4.exp_l9e1_4.exp_l9e4_4.exp_l9e6_4.exp_l10e2_4.exp_l10e6_4.exp_l10e3_4.exp_l10e7_4.exp_l11e4_4.exp_l11e2_4.exp_l11e7_4.exp_l11e3_4.exp_l12e3_4.exp_l12e1_4.exp_l12e5_4.exp_l12e2_4.exp_l13e4_4.exp_l13e6_4.exp_l13e3_4.exp_l13e7_4.exp_l14e0_4.exp_l14e6_4.exp_l14e3_4.exp_l14e5_4.exp_l15e0_4.exp_l15e4_4.exp_l15e2_4.exp_l15e6_4.exp_l16e7_4.exp_l16e1_4.exp_l16e5_4.exp_l16e6_4.exp_l17e6_4.exp_l17e2_4.exp_l17e7_4.exp_l17e5_4.exp_l18e0_4.exp_l18e2_4.exp_l18e7_4.exp_l18e4_4.exp_l19e0_4.exp_l19e5_4.exp_l19e7_4.exp_l19e4_4.exp_l20e6_4.exp_l20e7_4.exp_l20e2_4.exp_l20e4_4.exp_l21e3_4.exp_l21e4_4.exp_l21e2_4.exp_l21e0_4.exp_l22e2_4.exp_l22e6_4.exp_l22e4_4.exp_l22e3_4.exp_l23e0_4.exp_l23e1_4.exp_l23e2_4.exp_l23e7_4.exp_l24e4_4.exp_l24e1_4.exp_l24e3_4.exp_l24e7_4.exp_l25e1_4.exp_l25e0_4.exp_l25e3_4.exp_l25e6_4.exp_l26e1_4.exp_l26e6_4.exp_l26e2_4.exp_l26e4_4.exp_l27e5_4.exp_l27e6_4.exp_l27e1_4.exp_l27e4_4.exp_l28e2_4.exp_l28e4_4.exp_l28e3_4.exp_l28e0_4.exp_l29e2_4.exp_l29e5_4.exp_l29e0_4.exp_l29e7_4.exp_l30e7_4.exp_l30e0_4.exp_l30e4_4.exp_l30e2_4.exp_l31e1_4.exp_l31e2_4.exp_l31e0_4.exp_l31e7_4 \
      --bits_name main_2.attn_4.random4_per_layer_seed44


# 2.30 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4 \
      --bits_name main_2.attn_4.first4_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_28_4.layer_29_4.layer_30_4.layer_31_4 \
      --bits_name main_2.attn_4.last4_blocks

# 2.54 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_0_4.layer_1_4.layer_2_4.layer_3_4.layer_4_4.layer_5_4.layer_6_4.layer_7_4 \
      --bits_name main_2.attn_4.first8_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_24_4.layer_25_4.layer_26_4.layer_27_4.layer_28_4.layer_29_4.layer_30_4.layer_31_4 \
      --bits_name main_2.attn_4.last8_blocks

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4 \
      --bits_name main_2.attn_4

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_8 \
      --bits_name main_2.attn_8

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l6e0_4.exp_l16e5_4.exp_l19e3_4.exp_l22e6_4.exp_l0e2_4.exp_l6e2_4.exp_l24e2_4.exp_l11e6_4 \
      --bits_name main_2.attn_2.random_8_experts_seed42

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l4e0_4.exp_l10e0_4.exp_l3e6_4.exp_l7e1_4.exp_l8e5_4.exp_l23e6_4.exp_l24e6_4.exp_l16e5_4 \
      --bits_name main_2.attn_2.random_8_experts_seed43

DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l20e5_4.exp_l31e4_4.exp_l21e2_4.exp_l31e3_4.exp_l5e7_4.exp_l4e5_4.exp_l10e7_4.exp_l28e4_4 \
      --bits_name main_2.attn_2.random_8_experts_seed44


DEBUG=0 CUDA_VISIBLE_DEVICES=0 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l6e0_4.exp_l16e5_4.exp_l19e3_4.exp_l22e6_4.exp_l0e2_4.exp_l6e2_4.exp_l24e2_4.exp_l11e6_4.exp_l1e7_4.exp_l18e7_4.exp_l15e7_4.exp_l31e5_4.exp_l21e3_4.exp_l20e6_4.exp_l0e6_4.exp_l21e7_4.exp_l19e7_4.exp_l20e7_4.exp_l0e7_4.exp_l27e1_4.exp_l3e7_4.exp_l24e7_4.exp_l26e5_4 \
      --bits_name main_2.attn_2.random_23_experts_seed42

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l4e0_4.exp_l10e0_4.exp_l3e6_4.exp_l7e1_4.exp_l8e5_4.exp_l23e6_4.exp_l24e6_4.exp_l16e5_4.exp_l25e5_4.exp_l26e4_4.exp_l13e7_4.exp_l8e6_4.exp_l2e1_4.exp_l1e3_4.exp_l0e7_4.exp_l11e6_4.exp_l28e6_4.exp_l16e4_4.exp_l21e2_4.exp_l16e5_4.exp_l27e2_4.exp_l0e7_4.exp_l26e6_4 \
      --bits_name main_2.attn_2.random_23_experts_seed43

DEBUG=0 CUDA_VISIBLE_DEVICES=3 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_2.exp_l20e5_4.exp_l31e4_4.exp_l21e2_4.exp_l31e3_4.exp_l5e7_4.exp_l4e5_4.exp_l10e7_4.exp_l28e4_4.exp_l22e1_4.exp_l22e3_4.exp_l18e5_4.exp_l21e1_4.exp_l20e3_4.exp_l2e7_4.exp_l9e6_4.exp_l11e7_4.exp_l19e4_4.exp_l21e5_4.exp_l0e0_4.exp_l27e7_4.exp_l24e7_4.exp_l30e5_4.exp_l5e1_4 \
      --bits_name main_2.attn_2.random_23_experts_seed44


# 2.30 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=6 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4 \
      --bits_name main_2.attn_4.top4_cos_sim

# 2.54 bits
DEBUG=0 CUDA_VISIBLE_DEVICES=7 python quantize_gptq_mixtral.py \
      --model_name mistralai/Mixtral-8x7B-v0.1 \
      --bits main_2.attn_4.layer_31_4.layer_30_4.layer_12_4.layer_10_4.layer_13_4.layer_8_4.layer_29_4.layer_11_4 \
      --bits_name main_2.attn_4.top8_cos_sim