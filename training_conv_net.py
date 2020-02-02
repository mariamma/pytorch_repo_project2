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


class LeNet(nn.Module):

    def __init__(self, n_classes=10):
        emb_dim = 20
        '''
        Define the initialization function of LeNet, this function defines
        the basic structure of the neural network
        '''

        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.emb = nn.Linear(64*7*7, emb_dim)
        self.clf = nn.Linear(emb_dim, n_classes)

    def num_flat_features(self, x):
        '''
        Calculate the total tensor x feature amount
        '''

        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.emb(x)
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
        prob = model(img)
        loss = F.cross_entropy(prob, target)

        # backward propagation
        loss.backward()
        avg_loss.update(loss, img.shape[0])

        # Update the model parameters
        optimizer.step()

    return avg_loss.avg


if __name__ == "__main__":

    number_epochs = 100

    # Use torch.device("cuda:0") if you want to train on GPU
    # OR Use torch.device("cpu") if you want to train on CPU
    device = torch.device('cuda:0')

    model = LeNet(10).to(device)

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    trainloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    track_loss = []
    for i in tqdm(range(number_epochs)):
        loss = train_one_epoch(model, trainloader, optimizer, device)
        track_loss.append(loss)

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-ConvNet")
    plt.savefig("./img/training_convnet.jpg")

    torch.save(model.state_dict(), "./models/convNet.pt")
