export CUDA_VISIBLE_DEVICES=4

python probe_mixtral_bits.py \
    --bits_config_str main_2.exp_l0e0_8.exp_l1e3_8.exp_l2e0_8.exp_l3e4_8.exp_l4e5_8.exp_l5e6_8.exp_l6e4_8.exp_l7e0_8.exp_l8e1_8.exp_l9e7_8.exp_l10e4_8.exp_l11e5_8.exp_l12e2_8.exp_l13e0_8.exp_l14e2_8.exp_l15e3_8.exp_l16e0_8.exp_l17e6_8.exp_l18e0_8.exp_l19e6_8.exp_l20e3_8.exp_l21e6_8.exp_l22e3_8.exp_l23e3_8.exp_l24e2_8.exp_l25e2_8.exp_l26e0_8.exp_l27e4_8.exp_l28e2_8.exp_l29e3_8.exp_l30e0_8.exp_l31e0_8 \
    --expert_add_bits 8 \
    --target_average_bits 3

python probe_mixtral_bits.py \
    --bits_config_str main_2 \
    --expert_add_bits 8 \
    --target_average_bits 3