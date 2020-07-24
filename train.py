import torch
from coreset import update_fe

import time
import copy


def train_model(model, dataloaders, criterion, optimizer, num_epochs, task, indices, device):
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
