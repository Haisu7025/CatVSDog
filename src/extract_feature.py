# -*- coding:utf-8 -*-

import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image

img_to_tensor = transforms.ToTensor()


def make_model():
    resmodel = models.resnet18(pretrained=True)
    resmodel.load_state_dict(torch.load('trained_models/best_model.pth'))
    # resmodel.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return resmodel

# 特征提取


def extract_feature(resmodel, imgpath):
    num_frts = resmodel.fc.in_features
    resmodel.fc = torch.nn.Linear(num_frts, 2)
    resmodel.eval()

    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    # tensor = tensor.cuda()

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()

    return result_npy[0]


if __name__ == "__main__":
    model = make_model()
    cat_imgpath = 'data/cat.1.jpg'
    dog_imgpath = 'data/dog.1.jpg'
    print 'cat:{} \t dog:{}'.format(inference(model, cat_imgpath), inference(model, dog_imgpath))
    print 'cat:{} \n dog:{}'.format(extract_feature(model, cat_imgpath), extract_feature(model, dog_imgpath))
