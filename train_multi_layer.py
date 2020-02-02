"""
Starter Code in Pytorch for training a multi layer neural network.

** Takes around 30 minutes to train.
"""

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

from utils import AverageMeter


"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Model - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


class MLP(nn.Module):

    def __init__(self, n_classes=10):
        '''
        Define the initialization function of LeNet, this function defines
        the basic structure of the neural network
        '''

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


def train_one_epoch(model, trainloader, optimizer, device):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer, 
    """

    model.train()
    avg_loss = AverageMeter("average-loss")
    for batch_idx, (img, target) in enumerate(trainloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        out = model(img)
        loss = F.cross_entropy(out, target)

        # backward propagation
        loss.backward()
        avg_loss.update(loss, img.shape[0])

        # Update the model parameters
        optimizer.step()

    return avg_loss.avg


if __name__ == "__main__":

    number_epochs = 100

    device = torch.device('cpu')  # Replace with torch.device("cuda:0") if you want to train on GPU

    model = MLP(10).to(device)

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    trainloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    track_loss = []
    for i in tqdm(range(number_epochs)):
        loss = train_one_epoch(model, trainloader, optimizer, device)
        track_loss.append(loss)

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-MLP")
    plt.savefig("./img/training_mlp.jpg")

    torch.save(model.state_dict(), "./models/MLP.pt")
