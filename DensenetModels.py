import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.self_classifier = nn.Sequential(nn.Linear(kernelCount, 14))

    def gradients(self,outputs, inputs):
        a = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),create_graph=True)
        return a

    def forward(self, x,flag=None):
        u = self.densenet121(x)
        if flag=="pde":
            u_g = self.gradients(u, x)[0]
            u_x, u_y = u_g[:, 0], u_g[:, 1]
            u_xx = self.gradients(u_x, x)[0][:, 0]
            u_yy = self.gradients(u_y, x)[0][:, 1]
            X, Y = x[:, 0], x[:, 1]
            loss = (u_xx + u_yy) - torch.exp(-X) * (X - 2 + Y ** 3 + 6 * Y)
            return u, (loss ** 2).mean()
        else:
            return u

