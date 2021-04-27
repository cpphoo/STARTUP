import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
# import time
import os
# import glob

import configs
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args
from datasets import miniImageNet_few_shot, tiered_ImageNet_few_shot, ImageNet_few_shot

import utils
import wandb

from tqdm import tqdm
import random


def train(base_loader, model, optimization, start_epoch, stop_epoch, params, logger):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')     

    for epoch in tqdm(range(start_epoch,stop_epoch)):
        model.train()
        perf = model.train_loop(epoch, base_loader, optimizer, logger) 

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(), 
                        'optimizer': optimizer.state_dict()}, outfile)

        wandb.log({'loss': perf['Loss/avg']}, step=epoch+1)
        wandb.log({'top1': perf['top1/avg'],
                'top5': perf['top5/avg'],
                'top1_per_class': perf['top1_per_class/avg'],
                'top5_per_class': perf['top5_per_class/avg']}, step=epoch+1)
        
    return model

if __name__=='__main__':
    params = parse_args('train')
    image_size = 224
    bsize = params.bsize
    optimization = 'Adam'

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(params.seed)
    torch.random.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    random.seed(params.seed)

    save_dir = configs.save_dir
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s' % (
        save_dir, params.dataset, params.model, params.method, bsize)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (
            params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    logger = utils.create_logger(os.path.join(params.checkpoint_dir, 'checkpoint.log'), __name__)

    if params.method in ['baseline'] :

        if params.dataset == "miniImageNet":
            # Original Batchsize is 16
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=bsize, split=params.subset_split)
            base_loader = datamgr.get_data_loader(aug=params.train_aug, num_workers=8)
            params.num_classes = 64
        elif params.dataset == 'tiered_ImageNet':
            image_size = 84
            # Do no augmentation for tiered imagenet to be consisitent with the literature
            datamgr = tiered_ImageNet_few_shot.SimpleDataManager(
                image_size, batch_size=bsize, split=params.subset_split)
            base_loader = datamgr.get_data_loader(
                aug=False, num_workers=8)
            print("Number of images", len(base_loader.dataset))
            params.num_classes = 351
        elif params.dataset == 'ImageNet':
            datamgr = ImageNet_few_shot.SimpleDataManager(
                image_size, batch_size=bsize, split=params.subset_split)
            base_loader = datamgr.get_data_loader(
                aug=params.train_aug, num_workers=8)
            print("Number of images", len(base_loader.dataset))
            params.num_classes = 1000
        else:
           raise ValueError('Unknown dataset')

        model = BaselineTrain(model_dict[params.model], params.num_classes)

    elif params.method in ['protonet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if params.dataset == "miniImageNet":

            datamgr            = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            base_loader        = datamgr.get_data_loader(aug = params.train_aug)

        else:
           raise ValueError('Unknown dataset')

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
       
    else:
       raise ValueError('Unknown method')

    for arg in vars(params):
        logger.info(f"{arg}: {getattr(params, arg)}")

    logger.info(f"Image_size: {image_size}")
    logger.info(f"Optimization: {optimization}")

    wandb.init(project='cross_task_distillation',
               group=__file__,
               name=f'{__file__}_{params.checkpoint_dir}')

    wandb.config.update(params)

    model = model.cuda()

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params, logger=logger)
