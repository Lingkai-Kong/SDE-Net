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
import os
import argparse
import models
import data_loader

parser = argparse.ArgumentParser(description='PyTorch mc-dropout Training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--dataset', default='svhn', help='in domain dataset')
parser.add_argument('--eva_iter', default=10, type=int, help='number of passes when evaluation')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[20, 40], nargs='+', help='decreasing strategy')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


torch.manual_seed(0)

if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(0)

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getDataSet(args.dataset, args.batch_size, args.test_batch_size, args.imageSize)


# Model
print('==> Building model..')
net = models.Resnet_dropout()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train epoch:{} \tLoss: {:.6f} | Acc: {} ({}/{})'
        .format(epoch, train_loss/(len(train_loader)), 100.*correct/total, correct, total))

def test(epoch):
    net.eval()
    net.apply(apply_dropout)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = 0
            for j in range(args.eva_iter):
                outputs = outputs + F.softmax(net(inputs), dim=1)
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
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.droprate


if not os.path.isdir('./save_resnet_dropout_svhn'):
    os.makedirs('./save_resnet_dropout_svhn')
torch.save(net.state_dict(),'./save_resnet_dropout_svhn/final_model')


