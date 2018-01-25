# -*- coding:utf-8 -*-

import torch
import datasets
import torchvision.models as models
import time
import logger as lg
import evaluate
import torch.optim
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

opt = {
    'batch_size': 50,
    'cuda': True,
    'lr': 0.001,
    'momentum': 0.9,
    'lr_decay': 0.7,
    'weight_decay': 0.0001,
    'max_epochs': 2000,
    'lr_decay_epoch':50,
    'save_freq':10,
}

logger = lg.init_logger('train')
# for batch_x, batch_y in enumerate(myLoader):
#     print batch_x
#     print batch_y[0]
#     break


def train(train_loader, model, scheduler, criterion, optimizer, epoch):
    global opt
    global logger

    # training mode
    model.train()

    losses = 0
    acc = 0.0

    for batch_x, batch_y in enumerate(train_loader):
        scheduler.step()

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
        if criterion is None:
            loss = torch.nn.functional.binary_cross_entropy(res, label_var)
        else:
            loss = criterion(res, label_var)

        loss_d = loss.data.cpu().numpy()
        losses += loss_d

        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_x % 300 == 0 and batch_x != 0:
            losses = losses / 300
            acc = evaluate.eval(model)
            log_str = 'Epoch: [{0}][{1}/{2}]\t Loss {3} Avg Loss {4} Accuracy {5}'.format(
                epoch, batch_x, len(train_loader), loss.data.cpu().numpy(), losses, acc)
            losses = 0
            logger.info(log_str)     


def main():
    global opt
    global logger

    myTransforms = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    myDataset = datasets.CatDogDataset(
        transform=myTransforms,
        train=True)
    myLoader = data.DataLoader(
        dataset=myDataset,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=2,
    )
    print '===================BATCH:',opt['batch_size'],'======================'

    print '===================LENGTH:',len(myLoader),'======================'

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # model = Inception3.Inception3(aux_logits=False)
    # init model
    init_model = ''
    start_epoch = 0
    if init_model != '':
        logger.info('Loading pre-trained model from {0}!'.format(init_model))
        model.load_state_dict(torch.load(init_model))
        start_epoch = 1010
    criterion = torch.nn.CrossEntropyLoss()

    if opt['cuda']:
        logger.info('Using GPU to Shift the Calculation!')
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(start_epoch, opt['max_epochs']):
        # train for one epoch
        train(myLoader, model, scheduler, criterion, optimizer, epoch)
        if (epoch + 1) % opt['save_freq'] == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format('checkpoints', model.__class__.__name__, epoch + 1)
            torch.save(model.state_dict(), path_checkpoint)
        # LR_Policy(optimizer, opt['lr'], lambda_lr(epoch))


if __name__ == '__main__':
    main()