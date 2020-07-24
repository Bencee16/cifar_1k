import torch
import torch.nn as nn
import torch.optim as optim

import datetime
import time
import argparse

from utils import save_model
from coreset import create_coreset
from model_initialization import initialize_model, set_params_to_update
from train import train_model
from dataloader import load_data
from unit_tests import forgetting_test, shape_test

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs_proxy', type=int, default=20)
parser.add_argument('--num_epochs_core', type=int, default=150)
parser.add_argument('--model_type_proxy', default="resnet18")
parser.add_argument('--model_type_core', default="resnet50")
parser.add_argument('--use_pretrained_proxy', type=bool, default=True)
parser.add_argument('--use_pretrained_core', type=bool, default=True)
parser.add_argument('--freeze_weights_proxy', type=bool, default=False)
parser.add_argument('--freeze_weights_core', type=bool, default=True)
parser.add_argument('--coreset_size', type=int, default=1000)
parser.add_argument('--coreset_selection_method', default="reverse_forgetting_events")
parser.add_argument('--saving', type=bool, default=True)
parser.add_argument('--continue_from_selection', type=bool, default=False)
parser.add_argument('--dropout', type=bool, default=False)



def main(args):

    PATH = './'
    # path to forgetting stats coming from the ResNet18 model
    forgetting_path = "preloaded_forgetting/forgetting_stats"

    batch_size = args.batch_size
    print(batch_size)
    num_epochs = {'proxy': args.num_epochs_proxy, 'core': args.num_epochs_core}
    model_type = {'proxy': args.model_type_proxy, 'core': args.model_type_core}
    use_pretrained = {'proxy': args.use_pretrained_proxy, 'core': args.use_pretrained_core}
    freeze_weights = {'proxy': args.freeze_weights_proxy, 'core': args.freeze_weights_core}
    coreset_size = args.coreset_size
    coreset_selection_method = args.coreset_selection_method
    saving = args.saving
    continue_from_selection = args.continue_from_selection
    dropout = args.dropout

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

    indices = {'train': list(range(45000)),
               'val': list(range(45000, 50000))}

    trainset, valset, testset, dataloaders = load_data(batch_size, indices)


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
      print('starting (proxy) model training on the full data: \n')
      proxy_model, acc_history, forgetting_stats = train_model(proxy_model, dataloaders, criterion, optimizer, num_epochs[task], task, indices, device)
      times[task] = time.time() - start_time
      if saving:
          save_model(proxy_model, acc_history, forgetting_stats, modelname, PATH, config, 'proxy', times)


    # if continue_from_selection=True only core-training happens
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
    core_model, acc_history, _ = train_model(core_model, dataloaders, criterion, optimizer, num_epochs[task], task, indices, device)

    times[task] = time.time() - core_start_time
    if saving:
        save_model(core_model, acc_history, _, modelname, PATH, config, 'core', times)

    full_time = round(sum([v for (_, v) in times.items()]))
    print(f'The end-to-end training took {full_time//3600}h, {(full_time %3600)// 60}m, {full_time % 60}s')



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)