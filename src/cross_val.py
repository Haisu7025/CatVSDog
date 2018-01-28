# -*- coding:utf-8 -*-

import os
import shutil
import time
import logger
import torch
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torch.autograd import Variable
from torchvision import models
from torch.optim import lr_scheduler


log = logger.init_logger('cross_val')
part = True


def part_data(num=10):
    dog_file_list = range(10000)
    cat_file_list = range(10000)
    random.shuffle(dog_file_list)
    random.shuffle(cat_file_list)

    for i in range(10):
        os.mkdir('data/{}'.format(i))
        os.mkdir('data/{}/cat'.format(i))
        os.mkdir('data/{}/dog'.format(i))

        for file_index in cat_file_list[i * 1000:(i + 1) * 1000]:
            shutil.move('data/cat.{}.jpg'.format(file_index),
                        'data/{}/cat'.format(i))
        for file_index in dog_file_list[i * 1000:(i + 1) * 1000]:
            shutil.move('data/dog.{}.jpg'.format(file_index),
                        'data/{}/dog'.format(i))


def prep_model(init_model=None):
    model = models.resnet18(pretrained=False)
    num_frts = model.fc.in_features
    model.fc = torch.nn.Linear(num_frts, 2)
    if init_model is None:
        return model
    model.load_state_dict(torch.load(init_model))
    return model


def cross_val(val_index, dataloaders, dataset_sizes, model, criterion, optimizer):
    stime = time.time()

    # training phase
    running_corrects = 0.0
    for i in range(10):
        if i == val_index:
            continue
        else:
            model.train()
        for batch_x, batch_y in enumerate(dataloaders[i]):
            inputs, labels = batch_y
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_corrects += torch.sum(preds == labels.data)

    train_acc = running_corrects / (dataset_sizes[0] * 9)
    log.info('Validation Index [{}]/[{}] \t Training Phase Accuracy: {:.4f}'.format(
        val_index, 10, train_acc))

    # validation phase
    running_corrects = 0.0
    for batch_x, batch_y in enumerate(dataloaders[val_index]):
        inputs, labels = batch_y
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_corrects += torch.sum(preds == labels.data)

    train_acc = running_corrects / dataset_sizes[0]
    log.info('Validation Index [{}]/[{}] \t Validation Phase Accuracy: {:.4f}'.format(
        val_index, 10, train_acc))


def main():

    if not part:
        part_data()

    data_dir = 'data'
    data_transforms = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_datasets = {x: datasets.ImageFolder('data/{}'.format(x), data_transforms) for x in range(10)}
    dataloaders = {x: data.DataLoader(
        image_datasets[x], batch_size=10, shuffle=True, num_workers=2) for x in range(10)}
    dataset_sizes = {x: len(image_datasets[x]) for x in range(10)}

    model = prep_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for val_index in range(10):
        model = prep_model('trained_models/best_model.pth')
        if torch.cuda.is_available():
            model = model.cuda()
            # criterion = criterion.cuda()
        cross_val(val_index, dataloaders, dataset_sizes,
                  model, criterion, optimizer)


if __name__ == '__main__':
    main()
