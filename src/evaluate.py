# -*- coding:utf-8 -*-

import torch
import datasets
import models
import time
import torch.optim
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable

opt = {
    'batch_size': 50,
    'cuda': False,
    'lr': 0.001,
    'momentum': 0.9,
    'lr_decay': 0.7,
    'weight_decay': 0.0001,
    'init_model': ''
}


def test(test_loader, model):
    global opt

    # evaluate mode
    model.eval()
    for batch_x, batch_y in enumerate(test_loader):
        img = batch_y[:][0]
        label = batch_y[:][1]

        # transpose
        torch.transpose(img, 1, 2)
        torch.transpose(img, 2, 3)

        # Variable
        img_var = Variable(img)

        # cuda?
        if opt['cuda']:
            img_var = img_var.cuda()

        # forward
        res = model(img_var)

        # calculate accuracy
        acc = 0.0
        for i in range(opt['batch_size']):
            if res[i] == label[i]:
                acc += 1.0

        acc = acc / opt['batch_size']

        print 'Testing Accuracy:{.3f}'.format(acc)
        return acc


def eval(trained_model=''):
    global opt

    myTransforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    myDataset = datasets.CatDogDataset(
        num=9999,
        transform=myTransforms,
        train=False)
    myLoader = data.DataLoader(
        dataset=myDataset,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=2,
    )

    model = models.Inception3(aux_logits=False)
    if trained_model != '':
        model.load_state_dict(torch.load(trained_model))

    if opt['cuda']:
        print 'Using GPU to Shift the Calculation!'
        model = model.cuda()

    test(myLoader, model)


def main():
    global opt

    myTransforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    myDataset = datasets.CatDogDataset(
        num=9999,
        transform=myTransforms,
        train=False)
    myLoader = data.DataLoader(
        dataset=myDataset,
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=2,
    )

    model = models.Inception3(aux_logits=False)

    if opt['init_model'] != '':
        print 'Load model:', opt['init_model']
        model.load_state_dict(opt['init_model'])

    if opt['cuda']:
        print 'Using GPU to Shift the Calculation!'
        model = model.cuda()

    avg_acc = 0.0
    max_acc = 0.0
    for epoch in range(100 / opt['batch_size']):
        acc = test(myLoader, model)
        avg_acc += acc
        if acc > max_acc:
            max_acc = acc
    print 'Average accuracy:{.3f}, Maximum accuracy:{.3f}'.format(
        avg_acc, max_acc)


if __name__ == '__main__':
    main()
