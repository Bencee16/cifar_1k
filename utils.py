import torch
import os
import json


def create_folder(path, modelname, config):
    if 'models' not in os.listdir(path):
        os.mkdir(path+'models')
    if modelname not in os.listdir(path+'models/'):
        os.mkdir(path + 'models/' + modelname)
        with open(path + 'models/' + modelname + '/config.txt', 'w') as outfile:
            json.dump(config, outfile)


def save_model(model, acc_history, forgetting_stats, modelname, path, config, task, times):
    # saving config, model, accuracy, history, forgetting_stats
    print('Saving results')
    create_folder(path, modelname, config)
    path = path + 'models/'
    if task == 'proxy':
        torch.save(forgetting_stats, path + modelname + '/forgetting_stats')
    torch.save(acc_history, path + modelname + '/' + task + '_acc_history')
    torch.save(model.state_dict(), path + modelname + '/' + task + '_model.pth')
    with open(f'{path + modelname}/{task}_test_accuracy.txt', 'w') as f:
        f.write(str(acc_history['test']))
    with open(f'{path + modelname}/times.txt', 'w') as f:
        json.dump(times, f)


