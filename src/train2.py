# -*- coding:utf-8 -*-

import os
import time
import logger
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from torch.autograd import Variable
from torchvision import models
from torch.optim import lr_scheduler


log = logger.init_logger('resnet')

def train(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs = 25):
    stime = time.time()
    
    best_model_state = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0

            for batch_x, batch_y in enumerate(dataloaders[phase]):
                inputs, labels = batch_y

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if (batch_x + 1) % 200 == 0:
                    log.info('{} --- Current batch:{}/{}  loss:{}'.format(phase, batch_x + 1, len(dataloaders[phase]), loss.data.cpu().numpy()))
            
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'val':
                log.info('Epoch [{}]/[{}] \t Loss: {} \t Accuracy: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state = model.state_dict()

            torch.save(model.state_dict(), 'checkpoints/{}_{}_state.pth'.format(epoch, phase))
            log.info('Save module: checkpoints/{}_{}_state.pth'.format(epoch, phase))
            
    time_elapsed = time.time() - stime
    log.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    log.info('Best validate accuracy: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_state)
    return model


def main():
    # data
    data_dir = 'data_'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=10, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # model
    model = models.resnet18(pretrained=True)
    num_frts = model.fc.in_features
    model.fc = torch.nn.Linear(num_frts, 2)

    # train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.is_available():
        model = model.cuda()
        # criterion = criterion.cuda()

    model = train(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler)
    torch.save(model.state_dict(), 'trained_models/best_model.pth')

if __name__ == '__main__':
    main()
