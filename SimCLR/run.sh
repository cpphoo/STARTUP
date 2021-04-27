#!/bin/bash

# bash script to train SimCLR representation
export CUDA_VISIBLE_DEVICES=1


# Before running the commands, please take care of the TODO appropriately
for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease" 
do
    # TODO: Please set the following argument appropriately 
    # --dir: directory to save the student representation. 
    # --model: backbone type (supports resnet10, resnet12 and resnet18)
    # --teacher_path: initialization for the representation. Remove if want to 
    #                 start training from scratch
    # E.g. the following commands trains a SimCLR representation (initialized using the weights specified at
    #      ../teacher_miniImageNet/logs_deterministic/checkpoints/miniImageNet/ResNet10_baseline_256_aug/399.tar) 
    #      The student representation is saved at SimCLR_miniImageNet/$target_testset\_unlabeled_20/checkpoint_best.pkl
    python SIMCLR.py \
    --dir SimCLR_miniImageNet/$target_testset\_unlabeled_20 \
    --target_dataset $target_testset \
    --image_size 224 \
    --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv \
    --bsize 256 \
    --epochs 1000 \
    --save_freq 50 \
    --print_freq 10 \
    --seed 1 \
    --wd 1e-4 \
    --num_workers 4 \
    --model resnet10 \
    --teacher_path ../teacher_miniImageNet/logs_deterministic/checkpoints/miniImageNet/ResNet10_baseline_256_aug/399.tar \
    --teacher_path_version 0 \
    --eval_freq 2 \
    --batch_validate \
    --resume_latest 
done