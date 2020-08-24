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
import torch.backends.cudnn as cudnn
import data_loader
import os
import argparse
import numpy as np
import models 

parser = argparse.ArgumentParser(description='PyTorch SDENet Training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--lr2', default=0.01, type=float, help='learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=float, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[20], nargs='+', help='decreasing strategy')
parser.add_argument('--decreasing_lr2', default=[], nargs='+', help='decreasing strategy')

args = parser.parse_args()
print(args)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

batch_size = 128
Iter = 3622
Iter_test = 403
target_scale = 10.939756
# Data
print('==> Preparing data..')



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
net = models.SDENet(4)
net = net.to(device)


real_label = 0
fake_label = 1

criterion = nn.BCELoss()

optimizer_F = optim.SGD([ {'params': net.downsampling_layers.parameters()}, {'params': net.drift.parameters()},
{'params': net.fc_layers.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)

optimizer_G = optim.SGD([ {'params': net.diffusion.parameters()}], lr=args.lr2, momentum=0.9, weight_decay=5e-4)

def nll_loss(y, mean, sigma):
    loss = torch.mean(torch.log(sigma**2)+(y-mean)**2/(sigma**2))
    return loss
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
    if epoch == 0:
        net.sigma = 0.1
    if epoch == 30:
        net.sigma = 0.5
    train_loss = 0
    train_loss_in = 0
    train_loss_out = 0
    total = 0
    for iternum in range(Iter):
        inputs, targets = load_training(iternum)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_F.zero_grad()
        mean, sigma = net(inputs)
        loss = nll_loss(targets, mean, sigma)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 100.)
        optimizer_F.step()
        train_loss += loss.item()

        label = torch.full((batch_size,1), real_label, device=device)
        optimizer_G.zero_grad()
        predict_in = net(inputs, training_diffusion=True)
        loss_in = criterion(predict_in, label)
        loss_in.backward()
        label.fill_(fake_label)

        inputs_out = 2*torch.randn(batch_size, 90, device = device)+inputs
        predict_out = net(inputs_out, training_diffusion=True)
        loss_out = criterion(predict_out, label)
        
        loss_out.backward()
        train_loss_out += loss_out.item()
        train_loss_in += loss_in.item()
        optimizer_G.step()
      
    print('Train epoch:{} \tLoss: {:.6f}| Loss_in: {:.6f}| Loss_out: {:.6f}'.format(epoch, train_loss/Iter, train_loss_in/Iter, train_loss_out/Iter))

def test(epoch):
    net.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for iternum in range(Iter_test):
            inputs, targets = load_test(iternum)
            inputs, targets = inputs.to(device), targets.to(device)
            current_mean = 0
            for i in range(10):
                mean, sigma = net(inputs)
                current_mean = current_mean + mean
            current_mean = current_mean/10
            loss = mse(targets, current_mean)*target_scale
            test_loss += loss.item()
    
    print('Test epoch:{} \tLoss: {:.6f}'.format(epoch, np.sqrt(test_loss/Iter_test)))
           



for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        for param_group in optimizer_F.param_groups:
            param_group['lr'] *= args.droprate

    if epoch in args.decreasing_lr2:
        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_sdenet_msd'):
    os.makedirs('./save_sdenet_msd')
torch.save(net.state_dict(),'./save_sdenet_msd/final_model')

