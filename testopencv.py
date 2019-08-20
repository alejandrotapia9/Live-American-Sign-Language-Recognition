# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:26:59 2019

@author: aleja
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import shutil
import os

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
            [transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    
    model=torch.load('EntireModel.pt')
    model.eval()

    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space')
    

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,200)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,200)
    
    cv2.namedWindow("test")
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            
            #del all files in the dir
            filelist = [ f for f in os.listdir("C:/Users/aleja/Documents/UEES/IA/sign-language-mnist/data/asl_alphabet_test2/1")]
            for f in filelist:                
                os.remove(os.path.join("C:/Users/aleja/Documents/UEES/IA/sign-language-mnist/data/asl_alphabet_test2/1", f))
                
            shutil.move("C:/Users/aleja/Documents/UEES/IA/sign-language-mnist/opencv_frame_{}.png".format(img_counter),
                        "C:/Users/aleja/Documents/UEES/IA/sign-language-mnist/data/asl_alphabet_test2/1/opencv_frame_{}.png".format(img_counter))
            testset = torchvision.datasets.ImageFolder(root='./data/asl_alphabet_test2',
                                                       transform=test_transforms)
            testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers=4, batch_size=1)
            dataiter = iter(testloader)
            
            
            images, labels = dataiter.next()
            
            imshow(torchvision.utils.make_grid(images))
            print(' '.join('%5s' % classes[labels[j]] for j in range(1)))
            
            #imshow(torchvision.utils.make_grid(images))
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))
            img_counter += 1
    cam.release()
    
    cv2.destroyAllWindows()