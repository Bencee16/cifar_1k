import torch
import torchvision.models as models
import torch.nn as nn


def set_parameter_requires_grad(model, freeze_weights):
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_type, num_classes, use_pretrained, freeze_weights, device):
    model = None

    if model_type == "resnet18":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    else:
        print("Invalid model type, exiting...")
        exit()

    model = model.to(device)

    return model



def set_params_to_update(model, freeze_weights):
    params_to_update = model.parameters()
    if freeze_weights:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
              pass
                # print("\t",name)
    return params_to_update



def count_trainable_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))



def create_coreset(forgetting_stats, trainset, batch_size, train_indices, device, coreset_size=1000):

    forgetting_events = forgetting_stats['forgetting_events']
    nr_correct = forgetting_stats['nr_correct']


    # we have to insure that forgetting events is the highest for samples that were never learnt
    inf_vec = torch.Tensor([torch.max(forgetting_events)+1] * len(train_indices)).to(device)
    never_learnt = (nr_correct ==0)

    forgetting_events_with_unlearnt = forgetting_events + (inf_vec * never_learnt)

    a = list(zip(train_indices, forgetting_events_with_unlearnt))
    a.sort(key=lambda x: x[1], reverse=True)

    coreset_idxs = [i[0] for i in a[:coreset_size]]
    coreset = torch.utils.data.Subset(trainset, coreset_idxs)

    coreloader = torch.utils.data.DataLoader(coreset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

    return coreset, coreloader

