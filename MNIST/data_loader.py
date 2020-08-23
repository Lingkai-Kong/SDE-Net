import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#import os




def getMNIST(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))

    transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = []
    train_loader = DataLoader(
        datasets.MNIST(root='../data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.MNIST(root='../data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )
    ds.append(test_loader)

    return ds



def getSVHN(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []

    train_loader = DataLoader(
        datasets.SVHN(
            root='../data/svhn', split='train', download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform,
        ),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.SVHN(
            root='../data/svhn', split='test', download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform
        ),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)
    return ds


def getSEMEION(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SEMEION data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        datasets.SEMEION(
            root='../data/semeion', download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        datasets.SEMEION(
            root='../data/semeion', download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds



def getDataSet(data_type, batch_size,test_batch_size, imageSize):
    if data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size, test_batch_size, imageSize)
    elif data_type == 'mnist':
        train_loader, test_loader = getMNIST(batch_size, test_batch_size, imageSize)
    elif data_type == 'semeion':
        train_loader, test_loader = getSEMEION(batch_size, test_batch_size, imageSize)
    return train_loader, test_loader



if __name__ == '__main__':
    train_loader, test_loader = getDataSet('cifar10', 256, 1000, 28)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(inputs.shape)
        print(targets.shape)
