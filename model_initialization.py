import torch
import torchvision.models as models
import torch.nn as nn


class MyModel(torch.nn.Module):
    # We initialize from a base model
    def __init__(self, base_model):
        super(MyModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)

        # We split output activations to 2 streams and clamp the last neuron from below and above
        predictions = out[:, :-1]
        ones = out[:, -1].clamp(min=1, max=1)

        return predictions, ones


def set_parameter_requires_grad(model, freeze_weights):
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_type, num_classes, use_pretrained, freeze_weights, dropout, device):
    model = None

    if model_type == "resnet18":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
    elif model_type == 'resnet50':
        """ Resnet50
        """
        model = models.resnet50(pretrained=use_pretrained)

    else:
        print("Invalid model type, exiting...")
        exit()

    set_parameter_requires_grad(model, freeze_weights)
    num_ftrs = model.fc.in_features
    if dropout:
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, num_classes + 1)
        )
    else:
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes + 1)
        )

    # We add the extra stream for ones
    extended_model = MyModel(model)

    extended_model = extended_model.to(device)

    return extended_model


def set_params_to_update(model, freeze_weights):
    params_to_update = model.parameters()
    if freeze_weights:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    return params_to_update


def count_trainable_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
