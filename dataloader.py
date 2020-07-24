import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, SequentialSampler


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform):
        if phase == 'train':
            self.cifar10 = torchvision.datasets.CIFAR10(root='./data',
                                                        download=True,
                                                        train=True,
                                                        transform=transform)
        elif phase == 'val':
            self.cifar10 = torchvision.datasets.CIFAR10(root='./data',
                                                        download=True,
                                                        train=True,
                                                        transform=transform)
        elif phase == 'test':
            self.cifar10 = torchvision.datasets.CIFAR10(root='./data',
                                                        download=True,
                                                        train=False,
                                                        transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


def load_data(batch_size, indices):
    # input_size is equal to input_size used training on ImageNet
    input_size = 224

    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(input_size),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    transform_eval = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    trainset = MyDataset('train', transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              sampler=SubsetRandomSampler(indices['train']),
                                              num_workers=2)

    valset = MyDataset('val', transform_eval)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(indices['val']),
                                            num_workers=2)

    testset = MyDataset('test', transform_eval)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)

    toyset = torch.utils.data.Subset(trainset, list(range(batch_size)))
    toyloader = torch.utils.data.DataLoader(toyset, batch_size=batch_size,
                                            sampler=SequentialSampler(toyset),
                                            num_workers=2)

    dataloaders = {"train": trainloader, "val": valloader, "test": testloader, "toy": toyloader}

    return trainset, valset, testset, dataloaders
