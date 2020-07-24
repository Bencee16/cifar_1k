import torch
import random


def update_fe(preds, labels, idxs, previous_status, forgetting_events, nr_correct):
    # updates forgetting events statistics during the training run

    is_correct = (preds == labels.data)
    correct_idxs = idxs[is_correct]
    nr_correct[correct_idxs] += 1
    is_forgotten = (is_correct < previous_status[idxs])
    forgetting_events[idxs[is_forgotten]] += 1
    previous_status[idxs] = is_correct.type(torch.float)
    return previous_status, forgetting_events, nr_correct


def create_coreset(forgetting_stats, trainset, batch_size, train_indices, device, coreset_selection_method, coreset_size=1000):
    # from existing forgetting statistics, selects a coreset based on the given coreset selection method
    # returns the coreset, the coresetloader and the indices of the coreset in the original training set

    forgetting_events = forgetting_stats['forgetting_events']
    nr_correct = forgetting_stats['nr_correct']

    # we have to ensure that forgetting events is the highest for samples that were never learnt
    inf_vec = torch.Tensor([torch.max(forgetting_events)+1] * len(train_indices)).to(device)
    never_learnt = (nr_correct ==0)
    forgetting_events_with_unlearnt = forgetting_events + (inf_vec * never_learnt)

    # We create a zipped list of (indices, forgetting_events) so we can sort them
    zipped_list = list(zip(train_indices, forgetting_events_with_unlearnt))

    if coreset_selection_method == 'random':
        random.shuffle(zipped_list)
    elif coreset_selection_method == 'forgetting_events':
        # never learnt and most forgotten images get into the coreset
        zipped_list.sort(key=lambda x: x[1], reverse=True)
    elif coreset_selection_method == 'reverse_forgetting_events':
        # least forgotten images get into the coreset
        zipped_list.sort(key=lambda x: x[1], reverse=False)

    coreset_idxs = [i[0] for i in zipped_list[:coreset_size]]
    coreset = torch.utils.data.Subset(trainset, coreset_idxs)

    coreloader = torch.utils.data.DataLoader(coreset, batch_size=batch_size, shuffle=True, num_workers=2)

    return coreset, coreloader, coreset_idxs
