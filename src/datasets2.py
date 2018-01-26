# -*- coding:utf-8 -*-
from torch.utils import data
from PIL import Image
import torch
import os
import numpy as np
import random

WIDTH = 224
HEIGHT = 224
DATA_LIST = ''


def prepare_datalist(num, prop):

    catl = [('cat.' + str(i) + '.jpg') for i in range(10000)]
    dogl = [('dog.' + str(i) + '.jpg') for i in range(10000)]
    random.shuffle(catl)
    random.shuffle(dogl)

    train_index = int(num * prop)
    trainl = catl[:train_index] + dogl[:train_index]
    testl = catl[train_index :] + dogl[train_index :]
    random.shuffle(trainl)
    random.shuffle(testl)

    # write train filelist
    t = open('filelists/train.txt', 'w')
    for s in trainl:
        t.write(s)
        t.write('\n')
    t.close()

    #write test filelist
    t = open('filelists/test.txt', 'w')
    for s in testl:
        t.write(s)
        t.write('\n')
    t.close()


class CatDogDataset(data.Dataset):
    # Customized Dataset for Kaggle Cat-Dog Challenge
    def __init__(self, transform, train=True):
        self.transform = transform
        self.train = train

        # prepare data and labels
        # read filelists
        t = open('filelists/train.txt')
        self.train_data_list = t.readlines()
        t.close()
        t = open('filelists/test.txt') 
        self.test_data_list = t.readlines()
        t.close()

    def __len__(self):
        # length of data
        if self.train:
            return len(self.train_data_list)
        else:
            return len(self.test_data_list)
        
    def __getitem__(self, idx):
        # get item
        if self.train == True:
            imgn= self.train_data_list[idx][:-1]
        else:
            imgn= self.test_data_list[idx][:-1]
        if 'cat' in imgn:
            label = np.array([0, 1])
        elif 'dog' in imgn:
            label = np.array([1, 0])
        img = Image.open('data/' + imgn).resize((WIDTH, HEIGHT))
        # img = torch.from_numpy(np.array(img))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(label)




if __name__ == '__main__':
    prepare_datalist(10000, 0.8)