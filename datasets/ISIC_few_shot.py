# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")

import configs

def identity(x): return x

class CustomDatasetFromImages(Dataset):
    def __init__(self, transform, target_transform=identity, csv_path= configs.ISIC_path + "/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv", \
        image_path = configs.ISIC_path + "/ISIC2018_Task3_Training_Input/", split=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
            target_transform: pytorch transforms for targets
            split: the filename of a csv containing a split for the data to be used. 
                    If None, then the full dataset is used. (Default: None)
        """
        self.img_path = image_path
        self.csv_path = csv_path

        # Transforms
        self.transform = transform
        self.target_transform = target_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self.labels = (self.labels != 0).argmax(axis=1)
        
        # Calculate len
        self.data_len = len(self.image_name)
        self.split = split

        if split is not None:
            print("Using Split: ", split)
            split = pd.read_csv(split)['img_path'].values
            # construct the index
            ind = np.concatenate([np.where(self.image_name == j)[0] for j in split])
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)
        # self.targets = self.labels

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        temp = Image.open(self.img_path +  single_image_name + ".jpg")
        img_as_img = temp.copy()
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return self.transform(img_as_img), self.target_transform(single_image_label)

    def __len__(self):
        return self.data_len



class SimpleDataset:
    def __init__(self, transform, target_transform=identity, split=None):
        self.transform = transform
        self.target_transform = target_transform
        self.d = CustomDatasetFromImages(transform=self.transform, target_transform=self.target_transform, split=split)


    def __getitem__(self, i):
        img, target = self.d[i]
        return img, target

    def __len__(self):
        return len(self.d)


class SetDataset:
    def __init__(self, batch_size, transform, split=None):
        self.transform = transform
        self.split = split
        self.d = CustomDatasetFromImages(transform=self.transform, split=split)

        self.cl_list = sorted(np.unique(self.d.labels).tolist())
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0,
                                  pin_memory = False)        
        for cl in self.cl_list:
            ind = np.where(np.array(self.d.labels) == cl)[0].tolist()
            sub_dataset = torch.utils.data.Subset(self.d, ind)
            self.sub_dataloader.append(torch.utils.data.DataLoader(
                sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

# class SubDataset:
#     def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
#         self.sub_meta = sub_meta
#         self.cl = cl 
#         self.transform = transform
#         self.target_transform = target_transform

#     def __getitem__(self,i):

#         img = self.transform(self.sub_meta[i])
#         target = self.target_transform(self.cl)
#         return img, target

#     def __len__(self):
#         return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop' or transform_type == 'RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type == 'Scale' or transform_type == 'Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter',
                              'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, split=None):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.split = split

    def get_data_loader(self, aug, num_workers=12): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform, split=self.split)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers=num_workers, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100, split=None):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)
        self.split = split

    def get_data_loader(self, aug, num_workers=12): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform, split=self.split)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = num_workers, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':

    train_few_shot_params   = dict(n_way = 5, n_support = 5) 
    base_datamgr            = SetDataManager(224, n_query = 16)
    base_loader             = base_datamgr.get_data_loader(aug = True)

    cnt = 1
    for i, (x, label) in enumerate(base_loader):
        if i < cnt:
            print(label.size())
        else:
            break
