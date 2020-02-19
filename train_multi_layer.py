import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms



"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Model - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


class MLP(nn.Module):

    def __init__(self, n_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)

        self.clf = nn.Linear(16, n_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.clf(x)

        return out

"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Training - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


def train_one_epoch(net, trainloader, optimizer, device):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer, 
    """
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print(i, input, labels)    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print(running_loss)
    return running_loss
            



if __name__ == "__main__":

    device = torch.device('cpu')  # Replace with torch.device("cuda:0") if you want to train on GPU
    number_epochs = 800
    net = MLP(10).to(device)

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    trainloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    track_loss = []
    for i in tqdm(range(number_epochs)):
        loss = train_one_epoch(net, trainloader, optimizer, device)
        print('[%5d] loss: %.3f' %
                  (i + 1, loss))
        track_loss.append(loss)

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-MLP")
    plt.savefig("./img/training_mlp2.png")

    torch.save(net.state_dict(), "./models/MLP2.pt")
