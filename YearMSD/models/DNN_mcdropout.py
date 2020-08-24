import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DNN(nn.Module):

    def __init__(self, dropout):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(90, 50)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, 50)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(50, 50)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(50, 50)
        self.dropout4 = nn.Dropout(dropout)
        self.fc5 = nn.Linear(50, 50)
        self.dropout5 = nn.Dropout(dropout)
        self.fc6 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        x = self.dropout5(F.relu(self.fc5(x)))
        x = self.fc6(x)
        mean = x[:,0]
        sigma = x[:,1]
        sigma = F.softplus(sigma)+1e-6
        return mean, sigma


#def test():
#    model = Resnet_dropout()
#    return model  
# 
#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#if __name__ == '__main__':
#    model = test()
#    num_params = count_parameters(model)
#    print(num_params)