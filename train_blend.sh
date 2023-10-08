#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/aidata/pengfei/datasets/BlendedMVS/"

python train_blendedmvs.py --dataset blendedmvs --batch_size 4 --epochs 8 \
--patchmatch_iteration 1 2 2 --patchmatch_range 6 4 2 \
--patchmatch_num_sample 8 8 16 --propagate_neighbors 0 9 9 --evaluate_neighbors 9 9 9 \
--patchmatch_interval_scale 0.005 0.0125 0.025 \
--lr 0.001 \
--trainpath=$MVS_TRAINING --trainlist lists/blendedmvs/train.txt --vallist lists/blendedmvs/val.txt \
--loadckpt ./checkpoints/model_000007.ckpt \
--logdir ./checkpoints $@
