import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import os

import models
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot, tiered_ImageNet_few_shot

from tqdm import tqdm
import pandas as pd
import argparse
import random
import copy
import warnings


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def finetune(novel_loader, params, n_shot): 

    print("Loading Model: ", params.embedding_load_path)
    if params.embedding_load_path_version == 0:
        state = torch.load(params.embedding_load_path)['state']
        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
        sd = state
    elif params.embedding_load_path_version == 1:
        sd = torch.load(params.embedding_load_path)

        if 'epoch' in sd:
            print("Model checkpointed at epoch: ", sd['epoch'])
        sd = sd['model']
    # elif params.embedding_load_path_version == 3:
    #     state = torch.load(params.embedding_load_path)
    #     print("Model checkpointed at epoch: ", state['epoch'])
    #     state = state['model']
    #     state_keys = list(state.keys())
    #     for _, key in enumerate(state_keys):
    #         if "module." in key:
    #             # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
    #             newkey = key.replace("module.", "")
    #             state[newkey] = state.pop(key)
    #         else:
    #             state.pop(key)
    #     sd = state
    else:
        raise ValueError("Invalid load path version!")

    if params.model == 'resnet10':
        pretrained_model_template = models.ResNet10()
        feature_dim = pretrained_model_template.final_feat_dim
    elif params.model == 'resnet12':
        pretrained_model_template = models.Resnet12(width=1, dropout=0.1)
        feature_dim = pretrained_model_template.output_size
    elif params.model == 'resnet18':
        pretrained_model_template = models.resnet18(remove_last_relu=False, 
                                        input_high_res=True)
        feature_dim = 512
    else:
        raise ValueError("Invalid model!")

    pretrained_model_template.load_state_dict(sd)

    n_query = params.n_query
    n_way = params.n_way
    n_support = n_shot

    acc_all = []

    for i, (x, y) in tqdm(enumerate(novel_loader)):

        pretrained_model = copy.deepcopy(pretrained_model_template)
        classifier = Classifier(feature_dim, params.n_way)

        pretrained_model.cuda()
        classifier.cuda()

        ###############################################################################################
        x = x.cuda()
        x_var = x

        assert len(torch.unique(y)) == n_way
    
        batch_size = 4
        support_size = n_way * n_support 
       
        y_a_i = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()

        # split into support and query
        x_b_i = x_var[:, n_support:,: ,: ,:].contiguous().view(n_way*n_query, *x.size()[2:]).cuda() 
        x_a_i = x_var[:, :n_support,: ,: ,:].contiguous().view(n_way*n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)

        if params.freeze_backbone:
            pretrained_model.eval()
            with torch.no_grad():
                f_a_i = pretrained_model(x_a_i)
        else:
            pretrained_model.train()

         ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        

        if not params.freeze_backbone:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=0.01)

        ###############################################################################################
        total_epoch = 100
        
        classifier.train()

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if not params.freeze_backbone:
                    delta_opt.zero_grad()
                    

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                y_batch = y_a_i[selected_id]

                if params.freeze_backbone:
                    output = f_a_i[selected_id]
                else:
                    z_batch = x_a_i[selected_id]
                    output = pretrained_model(z_batch)

                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()
                if not params.freeze_backbone:
                    delta_opt.step()

        pretrained_model.eval()
        classifier.eval()

        with torch.no_grad():
            output = pretrained_model(x_b_i)
            scores = classifier(output)
       
        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        # print (correct_this/ count_this *100)
        acc_all.append((correct_this/ count_this *100))

        if (i+1) % 100 == 0:
            acc_all_np = np.asarray(acc_all)
            acc_mean = np.mean(acc_all_np)
            acc_std = np.std(acc_all_np)
            print('Test Acc (%d episodes) = %4.2f%% +- %4.2f%%' %
                (len(acc_all),  acc_mean, 1.96 * acc_std/np.sqrt(len(acc_all))))
        
        ###############################################################################################

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %
          (len(acc_all),  acc_mean, 1.96 * acc_std/np.sqrt(len(acc_all))))

    return acc_all

def main(params):

    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    if params.target_dataset == 'ISIC':
        datamgr = ISIC_few_shot
    elif params.target_dataset == 'EuroSAT':
        datamgr = EuroSAT_few_shot
    elif params.target_dataset == 'CropDisease':
        datamgr = CropDisease_few_shot
    elif params.target_dataset == 'ChestX':
        datamgr = Chest_few_shot
    elif params.target_dataset == 'miniImageNet_test':
        datamgr = miniImageNet_few_shot
    elif params.target_dataset == 'tiered_ImageNet_test':
        if params.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        datamgr = tiered_ImageNet_few_shot
    else:
        raise ValueError("Invalid Dataset!")
    
    results = {}
    shot_done = []
    print(params.target_dataset)
    for shot in params.n_shot:
        print(f"{params.n_way}-way {shot}-shot")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(params.seed)
        torch.random.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        random.seed(params.seed)
        novel_loader = datamgr.SetDataManager(params.image_size, n_eposide=params.n_episode,
                                                n_query=params.n_query, n_way=params.n_way,
                                                n_support=shot, split=params.subset_split).get_data_loader(
                                                aug=params.train_aug)
        acc_all = finetune(novel_loader, params, n_shot=shot)
        results[shot] = acc_all
        shot_done.append(shot)

        if params.save_suffix is None:
            pd.DataFrame(results).to_csv(os.path.join(params.save_dir, 
                params.source_dataset + '_' + params.target_dataset + '_' + 
                str(params.n_way) + 'way' + '.csv'), index=False)
        else:
            pd.DataFrame(results).to_csv(os.path.join(params.save_dir, 
                params.source_dataset + '_' + params.target_dataset + '_' + 
                str(params.n_way) + 'way_' + params.save_suffix + '.csv'), index=False)

    return 

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='few-shot Evaluation script')
    parser.add_argument('--save_dir', default='.', type=str, help='Directory to save the result csv')
    parser.add_argument('--source_dataset', default='miniImageNet', help='source_dataset')
    parser.add_argument('--target_dataset', default='miniImagenet',
                        help='test target dataset')
    parser.add_argument('--subset_split', type=str,
                        help='path to the csv files that contains the split of the data')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')
    parser.add_argument('--n_way', default=5, type=int,
                        help='class num to classify for training')
    parser.add_argument('--n_shot', nargs='+', default=[5], type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_episode', default=600, type=int, 
                        help='Number of episodes')
    parser.add_argument('--n_query', default=15, type=int, 
                        help='Number of query examples per class')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')
    parser.add_argument('--model', default='resnet10',
                        help='backbone architecture')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze the backbone network for finetuning')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--embedding_load_path', type=str,
                        help='path to load embedding')
    parser.add_argument('--embedding_load_path_version', type=int, default=1, 
                        help='how to load the embedding')
    parser.add_argument('--save_suffix', type=str, help='suffix added to the csv file')

    
    params = parser.parse_args()
    main(params)
   
