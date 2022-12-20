CUDA_VISIBLE_DEVICES=2 python train.py \
--experiment_description "classification v1.0 fc spike" \
--epoch 100 \
--arch SNN_Darts_s2e0 \
--seed 12345 \
--layers 8 \
--drop_path_prob 0.0 \
--cutout \
--auxiliary \
--save "normal darts. 012-spike" \