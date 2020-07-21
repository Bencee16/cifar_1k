import torch


def shape_test(model, device):
    x = torch.randn(2,3,32,32).to(device)
    y = model(x)
    assert(y.size() == torch.Size([2, 10]))
    print("Output shape ok")


def forgetting_test(forgetting_stats, num_epochs):
    nr_correct = forgetting_stats['nr_correct']
    forgetting_events = forgetting_stats['forgetting_events']
    assert all(nr_correct <= num_epochs), 'the number of correct classification must not excees the number of epochs'
    assert all(forgetting_events <= float(num_epochs)/2), 'forgetting events can happen at most in every second epoch'
    assert all(nr_correct >= forgetting_events), 'at least as many correct prediction must happen as forgetting'
    assert all(nr_correct+forgetting_events <= num_epochs), 'the correct number of correct predictions and forgetting events must not exceed the number of epochs'
