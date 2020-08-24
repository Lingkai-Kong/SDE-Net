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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import models
import data_loader
import time

parser = argparse.ArgumentParser(description='PyTorch SDENet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr2', default=0.005, type=float, help='learning rate')
parser.add_argument('--training_out', action='store_false', default=True, help='training_with_out')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
parser.add_argument('--dataset_inDomain', default='svhn', help='in domain dataset')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=float, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[20, 40], nargs='+', help='decreasing strategy')
parser.add_argument('--decreasing_lr2', default=[10, 30], nargs='+', help='decreasing strategy')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# Data

torch.manual_seed(args.seed)
random.seed(args.seed)

if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)


print('load in-domain data: ',args.dataset_inDomain)
train_loader_inDomain, test_loader_inDomain = data_loader.getDataSet(args.dataset_inDomain, args.batch_size, args.test_batch_size, args.imageSize)

# Model
print('==> Building model..')
net = models.SDENet(layer_depth=6, num_classes=10, dim=64)
net = net.to(device)

real_label = 0
fake_label = 1

criterion = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()

optimizer_F = optim.SGD([ {'params': net.downsampling_layers.parameters()}, {'params': net.drift.parameters()},
{'params': net.fc_layers.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)

optimizer_G = optim.SGD([ {'params': net.diffusion.parameters()}], lr=args.lr2, momentum=0.9, weight_decay=5e-4)


# use a smaller sigma during training for stability
net.sigma = 5

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    train_loss_out = 0
    train_loss_in = 0

    ##training with in-domain data
    for batch_idx, (inputs, targets) in enumerate(train_loader_inDomain):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_F.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_F.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    #training with out-of-domain data
        if args.training_out:
            label = torch.full((args.batch_size,1), real_label, device=device)
            optimizer_G.zero_grad()
            predict_in = net(inputs, training_diffusion=True)
            loss_in = criterion2(predict_in, label)
            loss_in.backward()
            label.fill_(fake_label)
            inputs_out = 2*torch.randn(args.batch_size,1, args.imageSize, args.imageSize, device = device)+inputs
            predict_out = net(inputs_out, training_diffusion=True)
            loss_out = criterion2(predict_out, label)
            loss_out.backward()
            train_loss_out += loss_out.item()
            train_loss_in += loss_in.item()
            optimizer_G.step()

    print('Train epoch:{} \tLoss: {:.6f} | Loss_in: {:.6f}, Loss_out: {:.6f} | Acc: {:.6f} ({}/{})'
        .format(epoch, train_loss/(len(train_loader_inDomain)), train_loss_in/len(train_loader_inDomain), train_loss_out/len(train_loader_inDomain), 100.*correct/total, correct, total))

def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader_inDomain):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = 0
            for j in range(args.eva_iter):
                current_batch = net(inputs)
                outputs = outputs + F.softmax(current_batch, dim = 1)
            outputs = outputs/args.eva_iter

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test epoch: {}| Acc: {:.6f} ({}/{})'
        .format(epoch, 100.*correct/total, correct, total))



for epoch in range(0, args.epochs):
    train(epoch)
    test(epoch)
    if epoch in args.decreasing_lr:
        for param_group in optimizer_F.param_groups:
            param_group['lr'] *= args.droprate
    if epoch in args.decreasing_lr2:
        for param_group in optimizer_G.param_groups:
            param_group['lr'] *= args.droprate

if not os.path.isdir('./save_sdenet_svhn_{}'.format(args.seed)):
    os.makedirs('./save_sdenet_svhn_{}'.format(args.seed))
torch.save(net.state_dict(),'./save_sdenet_svhn_{}/final_model'.format(args.seed))

