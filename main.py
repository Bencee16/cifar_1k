import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np

import time
import copy

import torch.nn as nn
import torch.optim as optim

from tests import shape_test
from utils import initialize_model, set_params_to_update, count_trainable_parameters


input_size = 224
batch_size = 64
NUM_TRAIN = 49000
NUM_VAL = 1000
num_epochs = 5000
model_name = "resnet18"
feature_extract = False


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

indices = {'train': list(range(45000)),
           'val': list(range(45000, 50000))}


####################


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, phase):
        if phase == 'train':
            self.cifar10 = torchvision.datasets.CIFAR10(root='./data',
                                                        download=True,
                                                        train=True,
                                                        transform=transform_train)
        elif phase == 'val':
            self.cifar10 = torchvision.datasets.CIFAR10(root='./data',
                                                        download=True,
                                                        train=True,
                                                        transform=transform_eval)
        elif phase == 'test':
            self.cifar10 = torchvision.datasets.CIFAR10(root='./data',
                                                        download=True,
                                                        train=False,
                                                        transform=transform_eval)

    def __getitem__(self, index):
        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


trainset = MyDataset('train')
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          sampler=SubsetRandomSampler(indices['train']),
                                          num_workers=2)

valset = MyDataset('val')
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(indices['val']),
                                        num_workers=2)

testset = MyDataset('test')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2)

toyset = torch.utils.data.Subset(trainset, list(range(batch_size)))

toyloader = torch.utils.data.DataLoader(toyset, batch_size=batch_size,
                                        # sampler=SubsetRandomSampler(list(range(10))),
                                        sampler=SequentialSampler(toyset),
                                        num_workers=2)

dataloaders_dict = {"train": toyloader, "val": valloader, "test": testloader, "toy": toyloader}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            print('{} Loss: {:.4f} Acc: {:.4f} corrects: {}'.format(phase, epoch_loss, epoch_acc, running_corrects))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True).to(device)

print(model_name, " initialized")
shape_test(model_ft, device)
params_to_update = set_params_to_update(model_ft, feature_extract)
print("Number of params to train: ", count_trainable_parameters(model_ft))

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)


