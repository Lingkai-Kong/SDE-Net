import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os



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
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform
        ),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)
    return ds

def getCIFAR10(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        datasets.CIFAR10(
            root='../data/cifar10', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        datasets.CIFAR10(
            root='../data/cifar10', train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds




def getCIFAR100(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        datasets.CIFAR100(
            root='../data/cifar100', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        datasets.CIFAR100(
            root='../data/cifar100', train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds

def getLSUN(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building LSUN data loader with {} workers".format(num_workers))
    ds = []
    import os
    scriptdir = os.path.dirname(__file__)
    datadir = os.path.join(scriptdir,'data')
    test_loader = DataLoader(
        datasets.LSUN(
            datadir, classes='test',
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),)
    ds.append(test_loader)
    ds.append(test_loader)

    return ds






def getTIM(batch_size, test_batch_size, img_size, **kwargs):
    data_transforms = transforms.Compose([
            transforms.ToTensor(),])
 

    data_dir = 'data/tiny-imagenet-200/'

    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'),data_transforms)

    dataloaders = DataLoader(image_datasets, batch_size=test_batch_size,
                                             shuffle=True, drop_last=True, **kwargs)
    ds = [dataloaders, dataloaders]

    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
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
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size, test_batch_size, imageSize)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size, test_batch_size, imageSize)
    elif data_type == 'semeion':
        train_loader, test_loader = getSEMEION(batch_size, test_batch_size, imageSize)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size, test_batch_size, imageSize)
    elif data_type == 'lsun':
        train_loader, test_loader = getLSUN(batch_size, test_batch_size, imageSize)
    elif data_type == 'tim':
        train_loader, test_loader = getTIM(batch_size, test_batch_size, imageSize)
       
    return train_loader, test_loader

#def getNonTargetDataSet(data_type, batch_size, imageSize, dataroot):
#    if data_type == 'cifar10':
#        _, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)
#    elif data_type == 'svhn':
#        _, test_loader = getSVHN(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)
#    elif data_type == 'imagenet':
#        testsetout = datasets.ImageFolder(dataroot+"/Imagenet_resize", transform=transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor()]))
#        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
#    elif data_type == 'lsun':
#        testsetout = datasets.ImageFolder(dataroot+"/LSUN_resize", transform=transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor()]))
#        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)

    #return test_loader
#train_loader, test_loader=getDataSet('mnist', 256,1000, 28)

if __name__ == '__main__':
    train_loader, test_loader = getDataSet('cifar10', 256, 1000, 28)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(inputs.shape)
        print(targets.shape)