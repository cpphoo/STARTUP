#!/bin/bash

# convert a pretrained imagenet model from PyTorch to a weight format that will be used 
# for other experiments. 
# TODO: Set --save_dir to specify the directory to save the converted model weights. 
python convert_imagenet_weight.py --save_dir resnet18 --model resnet18