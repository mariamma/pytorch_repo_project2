import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
# from vis_utils import *
import random;
import math;
from torchvision.datasets import FashionMNIST
from matplotlib import pyplot as plt


"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Model - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""

num_epochs = 20;
batch_size = 100;
learning_rate = 0.001;

trans_img = transforms.Compose([transforms.ToTensor()])
train_dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Training - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


if __name__ == "__main__":
    #instance of the Conv Net
    cnn = CNN();
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);


    losses = [];
    num_epochs = 2;
    track_loss = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.float())
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item());
            
            if (i) % 100 == 0:
                track_loss.append(loss.item())
                print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
                    

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-ConvNet")
    plt.savefig("./img/training_convnet1.png")

    torch.save(cnn.state_dict(), "./models/convNet1.pt")
