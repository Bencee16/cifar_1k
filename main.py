import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler, SequentialSampler

import torch.nn as nn
import torch.optim as optim

import datetime
import time
import copy

from utils import save_model
from coreset import update_fe, create_coreset
from model_initialization import initialize_model, set_params_to_update
from tests import forgetting_test, shape_test

PATH = './'
# path to forgetting stats coming from the ResNet18 model
forgetting_path = "preloaded_forgetting/forgetting_stats"

input_size = 224
batch_size = 64
num_epochs = {'proxy': 1, 'core': 1}
model_type = {'proxy': "resnet18", 'core': "resnet50"}
use_pretrained = {'proxy': True, 'core': True}
freeze_weights = {'proxy': False, 'core': True}
coreset_size = 1000
coreset_selection_method = "reverse_forgetting_events"
saving = True
continue_from_selection = False
dropout = False

config = {
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'model_type': model_type,
    'use_pretrained': use_pretrained,
    'freeze_weights': freeze_weights,
    'core_size': coreset_size,
    'coreset_selection_method': coreset_selection_method,
    'continue_from_selection': continue_from_selection,
    'dropout': dropout
}

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
now = str(datetime.datetime.now())
time_string = now[:10] + '-' + now[11:13] + '-' + now[14:16]

modelname = '_'.join(
    str(x) for x in [model_type['proxy'], num_epochs['proxy'], model_type['core'], num_epochs['core'], time_string])

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
                                        sampler=SequentialSampler(toyset),
                                        num_workers=2)

dataloaders = {"train": toyloader, "val": valloader, "test": testloader, "toy": toyloader}


def train_model(model, dataloaders, criterion, optimizer, num_epochs, task):
    since = time.time()

    acc_history = {'train': [], 'val': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    nr_correct = torch.zeros(len(indices['train'])).to(device)
    previous_status = torch.zeros(len(indices['train'])).to(device)
    forgetting_events = torch.zeros(len(indices['train'])).to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, idxs in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs, ones = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':

                        # calculate forgetting events
                        if task == 'proxy':
                            previous_status, forgetting_events, nr_correct = update_fe(preds, labels, idxs,
                                                                                       previous_status,
                                                                                       forgetting_events, nr_correct)

                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(indices[phase])
            epoch_acc = running_corrects.double() / len(indices[phase])

            acc_history[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.3f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.3f}'.format(best_acc))
    print('\n')

    # load best model weights
    model.load_state_dict(best_model_wts)

    print("testing best model on the test set")
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels, _ in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, ones = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    acc_history['test'] = float(test_acc * 100)

    assert len(dataloaders['test'].dataset) == 10000
    print(f'{task} accuracy of the network on the 10000 test images: {round(float(100 * test_acc), 3)}% \n')

    forgetting_stats = {'nr_correct': nr_correct, 'forgetting_events': forgetting_events}

    return model, acc_history, forgetting_stats


# main script
task = 'proxy'
times = {}

if not continue_from_selection:
  start_time = time.time()

  # # initialize proxy model
  proxy_model = initialize_model(model_type[task], num_classes, use_pretrained[task], freeze_weights[task], dropout, device)
  params_to_update = set_params_to_update(proxy_model, freeze_weights[task])
  optimizer = optim.Adam(params_to_update, lr=0.001)
  criterion = nn.CrossEntropyLoss()

  #train and save proxy
  print('starting training the (proxy) model on the full data: \n')
  proxy_model, acc_history, forgetting_stats = train_model(proxy_model, dataloaders, criterion, optimizer, num_epochs[task], task)

  times[task] = time.time() - start_time

  if saving:
      save_model(proxy_model, acc_history, forgetting_stats, modelname, PATH, config, 'proxy', times)


# run this section for only core-training
if continue_from_selection:
    forgetting_stats = torch.load(PATH + forgetting_path, map_location=device)
    print('running coretraining using loaded forgetting stats \n')

#select coreset
selection_start_time = time.time()
# forgetting_test(forgetting_stats, num_epochs['proxy'])
print('selecting coreset \n')
coreset, coreloader, coreset_idxs = create_coreset(forgetting_stats, trainset, batch_size, indices['train'], device, coreset_selection_method, coreset_size=coreset_size)
print(f'{len(coreset)} sized coreset selected by method {coreset_selection_method} \n')
times['selection'] = time.time() - selection_start_time

#initialize core model
core_start_time = time.time()
task = 'core'
core_model = initialize_model(model_type[task], num_classes, use_pretrained[task], freeze_weights[task], dropout, device)
params_to_update = set_params_to_update(core_model, freeze_weights[task])
optimizer = optim.Adam(params_to_update, lr=0.001)
criterion = nn.CrossEntropyLoss()

# change dataloader in training phase to coreloader
dataloaders['train'] = coreloader
indices['train'] = coreset_idxs

# train core_model
core_model, acc_history, _ = train_model(core_model, dataloaders, criterion, optimizer, num_epochs['core'], 'core')

times[task] = time.time() - core_start_time
if saving:
    save_model(core_model, acc_history, _, modelname, PATH, config, 'core', times)

full_time = round(sum([v for (_, v) in times.items()]))
print(f'The end-to-end training took {full_time//3600}h, {(full_time %3600)// 60}m, {full_time % 60}s')

