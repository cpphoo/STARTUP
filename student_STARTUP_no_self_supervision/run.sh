#!/bin/bash

# bash script to train STARTUP representation without SimCLR self-supervision
export CUDA_VISIBLE_DEVICES=1

##############################################################################################
# Train student representation using MiniImageNet as the source 
##############################################################################################
# Before running the commands, please take care of the TODO appropriately
for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease" "miniImageNet_test" 
do  
    # TODO: Please set the following argument appropriately 
    # --teacher_path: filename for the teacher model
    # --base_path: path to find base dataset
    # --dir: directory to save the student representation. 
    # E.g. the following commands trains a STARTUP representation based on the teacher specified at
    #      ../teacher_miniImageNet/logs_deterministic/checkpoints/miniImageNet/ResNet10_baseline_256_aug/399.tar 
    #      The student representation is saved at miniImageNet_source/$target_testset\_unlabeled_20/checkpoint_best.pkl
    python STARTUP_no_SS.py \
    --dir miniImageNet_source/$target_testset\_unlabeled_20 \
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
    --base_dataset miniImageNet \
    --base_path /scratch/datasets/miniImageNet_full_resolution/train \
    --base_no_color_jitter \
    --base_val_ratio 0.05 \
    --eval_freq 2 \
    --resume_latest 
done


##############################################################################################
# Train student representation using ImageNet as the source 
##############################################################################################
# Before running the commands, please take care of the TODO appropriately

for target_testset in "ChestX" "ISIC" "EuroSAT" "CropDisease"
do  
    # TODO: Please set the following argument appropriately 
    # --teacher_path: filename for the teacher model
    # --base_path: path to find base dataset
    # --dir: directory to save the student representation. 
    # E.g. the following commands trains a STARTUP representation based on the teacher specified at
    #      ../teacher_ImageNet/resnet18/checkpoint.pkl 
    #      The student representation is saved at ImageNet_source/$target_testset\_unlabeled_20/checkpoint_best.pkl
    python STARTUP_no_SS.py \
    --dir ImageNet_source/$target_testset\_unlabeled_20 \
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
    --model resnet18 \
    --teacher_path ../teacher_ImageNet/resnet18/checkpoint.pkl \
    --teacher_path_version 1 \
    --base_dataset ImageNet \
    --base_path /scratch/datasets/imagenet/train \
    --base_no_color_jitter \
    --base_val_ratio 0.01 \
    --eval_freq 2 \
    --resume_latest 
done

##############################################################################################
# Train student representation using tieredImageNet as the source 
##############################################################################################
# Before running the commands, please take care of the TODO appropriately

for target_testset in "tiered_ImageNet_test" 
do
    # TODO: Please set the following argument appropriately 
    # --teacher_path: filename for the teacher model
    # --base_path: path to find base dataset
    # --dir: directory to save the student representation. 
    # --target_subset_split Either datasets/split_seed_1/$target\_unlabeled_10.csv (for the less unlabeled data setup) 
    #                      or datasets/split_seed_1/$target\_unlabeled_50.csv (for the more unlabeled data setup)
    # E.g. the following commands trains a STARTUP representation based on the teacher specified at
    #      ../teacher_miniImageNet/logs_deterministic/checkpoints/tiered_ImageNet/ResNet12_baseline_256/89.tar 
    #      The student representation is saved at tiered_ImageNet__source/$target_testset\_unlabeled_50/checkpoint_best.pkl
    python STARTUP_no_SS.py \
    --dir tiered_ImageNet_source/$target_testset\_unlabeled_50 \
    --target_dataset $target_testset \
    --image_size 84 \
    --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_50.csv \
    --bsize 256 \
    --epochs 100 \
    --save_freq 50 \
    --print_freq 10 \
    --seed 1 \
    --wd 1e-4 \
    --num_workers 2 \
    --model resnet12 \
    --teacher_path ../teacher_miniImageNet/logs_deterministic/checkpoints/tiered_ImageNet/ResNet12_baseline_256/89.tar \
    --teacher_path_version 0 \
    --base_dataset tiered_ImageNet \
    --base_path /scratch/datasets/tiered_imagenet/tiered_imagenet/original_split/train \
    --base_no_color_jitter \
    --base_val_ratio 0.05 \
    --eval_freq 2 \
    --resume_latest
done