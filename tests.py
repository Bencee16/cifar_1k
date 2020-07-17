import torch


def shape_test(model, device):
    x = torch.randn(2,3,32,32).to(device)
    y = model(x)
    assert(y.size() == torch.Size([2, 10]))
    print("Output shape ok")
