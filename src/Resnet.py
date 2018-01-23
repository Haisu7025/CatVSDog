import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        # Conv1
        self.conv1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # FC
        self.fc = nn.Sequential(
            nn.Linear(16 * 26 * 26,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,1)
        )

    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
    
    def forward(self, input):
        fwd = self.conv1(input)
        fwd = self.conv2(fwd)
        fwd = self.conv3(fwd)
        fwd = fwd.view(-1, 16 * 26 * 26)
        fwd = self.fc(fwd)
        fwd = F.sigmoid(fwd)
         
        return fwd


        