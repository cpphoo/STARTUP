import torchvision
import torchvision.models
import argparse

import models

import torch.nn as nn

import os
import torch

def main(args):
    if args.model == 'resnet18':
        backbone = models.resnet18(remove_last_relu=False, input_high_res=True).cuda()
    else:
        raise ValueError("Invalid backbone!")
    
    pretrained_model = getattr(
        torchvision.models, args.model)(pretrained=True).cuda()

    # load the backbone parameters
    for i in range(5):
        layer_name = f"layer{i}"

        # the first layer requires special handling
        # If the input is low resolution, then the first conv1 will have kernel size 3x3
        # instead of 7x7
        if layer_name == 'layer0':
            mod = getattr(backbone, layer_name)
            mod[0].load_state_dict(
                getattr(pretrained_model, 'conv1').state_dict())
            mod[1].load_state_dict(
                getattr(pretrained_model, 'bn1').state_dict())
        else:
            getattr(backbone, layer_name).load_state_dict(
                getattr(pretrained_model, layer_name).state_dict())

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    sd = {
        'model': backbone.state_dict(), 
        'clf': pretrained_model.fc.state_dict()
    }
    
    torch.save(sd, os.path.join(args.save_dir, 'checkpoint.pkl'))


    
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the pretrained ImageNet ResNet weight")
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the pretrained weights')
    parser.add_argument('--model', type=str, default='resnet18', help='which resnet model')
    args = parser.parse_args()
    main(args)    
