# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:04:24 2019

@author: aleja
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    
    trainset = torchvision.datasets.ImageFolder(root='./data/asl_alphabet_train',transform=transform)
    print(trainset.classes)
    trainloader = torch.utils.data.DataLoader(trainset,shuffle=True, num_workers=4, batch_size=4)
    
    testset = torchvision.datasets.ImageFolder(root='./data/asl_alphabet_test',transform=transform)
    print(testset.classes)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=4, batch_size=4)
    
    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')
    
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
   
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    
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

    
    
    net = Net()    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(1):  
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data           
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')
    
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth:th: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
      
    #Porcentaje de Error
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    print('Accuracy of the network on the test images: %d %%' %(100 * correct / total))
    
