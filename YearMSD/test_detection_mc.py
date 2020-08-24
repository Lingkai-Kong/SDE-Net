###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################
# Writer: Kimin Lee 
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import calculate_log as callog
import models
import math
import os 
from numpy.linalg import inv
import numpy as np
# Training settings
parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
parser.add_argument('--eval_iter', default=20, type=int, help='number of stochastic forward passes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--index', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--pre_trained_net', default='', help="path to pre trained_net")

args = parser.parse_args()
print(args)

outf = 'test/'+'mcdp'

if not os.path.isdir(outf):
    os.makedirs(outf)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)


print('Load model')

model = models.DNN(0.05)
model = model.to(device)
model.load_state_dict(torch.load(args.pre_trained_net))

X_train, y_train, X_test, y_test = data_loader.load_dataset('MSD')
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
Iter_test = 100
batch_size = 512
target_scale = 10.939756

X_out = data_loader.load_dataset('boston')
X_out = torch.from_numpy(X_out).type(torch.FloatTensor)




def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def mse(y, mean):
    loss = torch.mean((y-mean)**2)
    return loss

def load_test(iternum):
    x = X_test[iternum*batch_size:(iternum+1)*batch_size]
    y = y_test[iternum*batch_size:(iternum+1)*batch_size]
    return x, y

def generate_target(model):
    model = model.eval() 
    model.apply(apply_dropout) 
    test_loss = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt'%outf, 'w')

    with torch.no_grad():
        for iternum in range(Iter_test):
            data, targets = load_test(iternum)
            data, targets = data.to(device), targets.to(device)
            current_mean = 0
            temp = 0
            for j in range(args.eval_iter):
                mean, sigma = model(data)
                current_mean = mean + current_mean
                if j == 0:
                    Mean = torch.unsqueeze(mean,1)
                else:
                    Mean = torch.cat((Mean, torch.unsqueeze(mean,1)),dim=1)
            current_mean = current_mean/args.eval_iter

            Var = Mean.std(dim=1)
            loss = mse(targets, current_mean)*target_scale
            test_loss += loss.item()
            for i in range(data.size(0)):
                soft_out = Var[i].item()
                f1.write("{}\n".format(-soft_out))

    f1.close()

    print('\n Final RMSE: {}'.format(np.sqrt(test_loss/Iter_test)))

def generate_non_target(model):
    model = model.eval()
    model.apply(apply_dropout)

    f2 = open('%s/confidence_Base_Out.txt'%outf, 'w')
    with torch.no_grad():
   
        data  = X_out.to(device)
        current_mean = 0
        temp = 0
        for j in range(args.eval_iter):
            mean, sigma = model(data)
            if j == 0:
                Mean = torch.unsqueeze(mean,1)
            else:
                Mean = torch.cat((Mean, torch.unsqueeze(mean,1)),dim=1)

        Var = Mean.std(dim=1)
        for i in range(data.size(0)):
            soft_out = Var[i].item()
            f2.write("{}\n".format(-soft_out))
    f2.close()

print('generate log from in-distribution data')
generate_target(model)
print('generate log  from out-of-distribution data')
generate_non_target(model)
print('calculate metrics for OOD')
callog.metric(outf, 'OOD')
