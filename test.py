# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:51:23 2019

@author: aleja
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F

if __name__ == "__main__":
    
    class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 15, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(15, 29, 5)
                self.fc1 = nn.Linear(725, 180)
                self.fc2 = nn.Linear(180, 90)
                self.fc3 = nn.Linear(90, 29)
        
            def forward(self, x):
                in_size=x.size(0)
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(in_size,-1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    test_transforms = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    
    model=torch.load('EntireModel.pt')
    model.eval()
    
    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')
    
    testset = torchvision.datasets.ImageFolder(root='./data/asl_alphabet_test',transform=test_transforms)
    print(testset.classes)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=4, batch_size=4)
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    imshow(torchvision.utils.make_grid(images))
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
      