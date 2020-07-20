import torchvision.models as models
import torch.nn as nn


def set_parameter_requires_grad(model, freeze_weights):
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, use_pretrained, freeze_weights):
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_weights)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


# Question: do we need to return the model?
def set_params_to_update(model, feature_extract):

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)
    return params_to_update



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

