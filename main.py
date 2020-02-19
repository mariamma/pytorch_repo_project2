"""
Code to use the saved models for testing
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
from sklearn.metrics import confusion_matrix


def test(model, testloader):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer,
    """

    model.eval()
    avg_loss = AverageMeter("average-loss")

    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss.update(loss, img.shape[0])

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss.avg, y_gt, y_pred_label


def testML(cnn2,testloader):
    correct = 0
    total = 0
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = cnn2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # Append batch prediction results
            predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)    


if __name__ == "__main__":

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    from train_multi_layer import MLP
    model_MLP = MLP(10)
    model_MLP.load_state_dict(torch.load("./models/MLP2.pt"))

    from training_conv_net import CNN
    model_conv_net = CNN()
    model_conv_net.load_state_dict(torch.load("./models/model_cnn3.pth"))

    testML(model_MLP, testloader)
    loss, gt, pred = test(model_MLP, testloader)
    print("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
    with open("multi-layer-net1.txt", 'w') as f:
         f.write("Loss on Test Data : {}\n".format(loss))
         f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
         f.write("gt_label,pred_label \n")
         for idx in range(len(gt)):
             f.write("{},{}\n".format(gt[idx], pred[idx]))

    testML(model_conv_net, testloader)         
    loss, gt, pred = test(model_conv_net, testloader)
    print("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
    with open("convolution-neural-net1.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
