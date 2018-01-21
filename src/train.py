# -*- coding:utf-8 -*-

import torch
import datasets
import models
import time
import torch.optim
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy


opt = {
    'batch_size': 10,
    'cuda': False,
    'lr': 0.001,
    'momentum': 0.9,
    'lr_decay': 0.7,
    'weight_decay': 0.0001,
    'max_epochs': 2000
}


# for batch_x, batch_y in enumerate(myLoader):
#     print batch_x
#     print batch_y[0]
#     break


def train(train_loader, model, criterion, optimizer, epoch):
    global opt

    # training mode
    model.train()

    losses = 0

    for batch_x, batch_y in enumerate(train_loader):
        label = batch_y[:][1]
        label = label.numpy()
        label = torch.from_numpy(label).float()
        img = batch_y[:][0]

        # transpose
        torch.transpose(img, 1, 2)
        torch.transpose(img, 2, 3)

        # Variable
        label_var = Variable(label)
        img_var = Variable(img)

        # cuda?
        if opt['cuda']:
            label_var = label_var.cuda()
            img_var = img_var.cuda()

        # forward and backward
        res = model(img_var)
        # print '!!!', res
        # print '###', label
        loss = criterion(res, label_var)
        losses += loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_x % 50 == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Loss {3}'.format(
                epoch, batch_x, len(train_loader), losses)
            losses = 0
            print log_str


def main():
    global opt

    myTransforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    myDataset = datasets.CatDogDataset(
        num=9999,
        transform=myTransforms,
        train=True)
    myLoader = data.DataLoader(
        dataset=myDataset,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=2,
    )

    model = models.Inception3(aux_logits=False)
    criterion = torch.nn.MSELoss()

    if opt['cuda']:
        print 'Using GPU to Shift the Calculation!'
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), opt['lr'], momentum=0, weight_decay=opt['weight_decay'])

    def lambda_lr(epoch): return opt['lr_decay'] ** ((epoch + 1) //
                                                     opt['lr_decay_epoch'])  # poly policy

    for epoch in range(opt['max_epochs']):
        # train for one epoch
        train(myLoader, model, criterion, optimizer, epoch)

        LR_Policy(optimizer, opt['lr'], lambda_lr(epoch))


if __name__ == '__main__':
    main()
