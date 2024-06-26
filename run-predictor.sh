for id in $(seq 0 31)
do
    CUDA_VISIBLE_DEVICES=0 python run-mixtral-predictor.py --ffn_block_id=$id
done