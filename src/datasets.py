# -*- coding:utf-8 -*-
from torch.utils import data
from PIL import Image
import torch
import os
import numpy as np

WIDTH = 224
HEIGHT = 224


class CatDogDataset(data.Dataset):
    # Customized Dataset for Kaggle Cat-Dog Challenge
    def __init__(self, num, transform, train=True):
        self.num = num
        self.transform = transform
        self.train = train

        # prepare data and labels
        cat_data_list = [('data/cat.' + str(i) + '.jpg', np.array([1]))
                         for i in range(self.num)]
        dog_data_list = [('data/dog.' + str(i) + '.jpg', np.array([0]))
                         for i in range(self.num)]
        self.train_data_list = cat_data_list[:9949] + dog_data_list[:9949]
        self.test_data_list = cat_data_list[9950:] + dog_data_list[9950:]

    def __len__(self):
        # length of data
        if self.train:
            return len(self.train_data_list)
        else:
            return len(self.test_data_list)
        
    def __getitem__(self, idx):
        # get item
        if self.train == True:
            imgn, label = self.train_data_list[idx]
        else:
            imgn, label = self.test_data_list[idx]
        img = Image.open(imgn).resize((WIDTH, HEIGHT))
        # img = torch.from_numpy(np.array(img))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(label)
