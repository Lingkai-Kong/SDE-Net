#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:34:10 2019

@author: lingkaikong
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import models
import data_loader
import os
import argparse
import numpy as np

import torch.distributions.normal as normal
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=float, default=0)
parser.add_argument('--decreasing_lr', default=[20], nargs='+', help='decreasing strategy')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--dropout', type=float, default=0.05) #https://github.com/yaringal/DropoutUncertaintyExps/blob/a6259f1db8f5d3e2d743f88ecbde425a07b12445/YearPredictionMSD/net/net/net.py
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

batch_size = 128
Iter = 3622
Iter_test = 403
target_scale = 10.939756

torch.manual_seed(args.seed)


if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)



X_train, y_train, X_test, y_test = data_loader.load_dataset('MSD')

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)

# Model
print('==> Building model..')
dropout = args.dropout
net = models.DNN(dropout)
net = net.to(device)


tau = 0.0128363911266 # obtained from BO 
lengthscale = 1e-4
reg = lengthscale**2 * (1 - dropout) / (2. * len(X_train) * tau)

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=reg)




def calculate_loss(mean, std, target):
    likelihood = normal.Normal(mean, std).log_prob(target)
    return -likelihood.mean()


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def mse(y, mean):
    loss = torch.mean((y-mean)**2)
    return loss


def load_training(iternum):
    x = X_train[iternum*batch_size:(iternum+1)*batch_size]
    y = y_train[iternum*batch_size:(iternum+1)*batch_size]
    return x, y

def load_test(iternum):
    x = X_test[iternum*batch_size:(iternum+1)*batch_size]
    y = y_test[iternum*batch_size:(iternum+1)*batch_size]
    return x, y

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    for iternum in range(Iter):
        inputs, targets = load_training(iternum)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        mean, sigma = net(inputs)
        loss = calculate_loss(mean, sigma, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
      
    
    print('Train epoch:{} \tLoss: {:.6f}'.format(epoch, train_loss/Iter))

def test(epoch):
    net.eval()
    net.apply(apply_dropout)
    test_loss = 0
    total = 0
    with torch.no_grad():
        for iternum in range(Iter_test):
            inputs, targets = load_test(iternum)
            inputs, targets = inputs.to(device), targets.to(device)
            expected_mean = 0 
            for i in range(10):
                mean, sigma = net(inputs)
                expected_mean = mean + expected_mean
            expected_mean = expected_mean/10
            loss = mse(targets, expected_mean)*target_scale
            test_loss += loss.item()
            total += targets.size(0)
      
    
    print('Test epoch:{} \tLoss: {:.6f}'.format(epoch, np.sqrt(test_loss/Iter_test)))
           



for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_mc_msd'):
    os.makedirs('./save_mc_msd')
torch.save(net.state_dict(),'./save_mc_msd/final_model')

