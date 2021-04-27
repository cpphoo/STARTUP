# Compute accuracy 
import torch

def accuracy(logits, ground_truth, topk=[1, ]):
    assert len(logits) == len(ground_truth)
    # this function will calculate per class acc
    # average per class acc and acc

    n, d = logits.shape
    
    label_unique = torch.unique(ground_truth)
    acc = {}
    acc['average'] = torch.zeros(len(topk))
    acc['per_class_average'] = torch.zeros(len(topk))
    acc['per_class'] = [[] for _ in label_unique]
    acc['gt_unique'] = label_unique
    acc['topk'] = topk
    acc['num_classes'] = d

    max_k = max(topk)
    argsort = torch.argsort(logits, dim=1, descending=True)[:, :min([max_k, d])]
    correct = (argsort == ground_truth.view(-1, 1)).float()

    for indi, i in enumerate(label_unique):
        ind = torch.nonzero(ground_truth == i, as_tuple=False).view(-1)
        correct_target = correct[ind]

        # calculate topk 
        for indj, j in enumerate(topk):
            num_correct_partial = torch.sum(correct_target[:, :j]).item()
            acc_partial = num_correct_partial / len(correct_target)
            acc['average'][indj] += num_correct_partial
            acc['per_class_average'][indj] += acc_partial
            acc['per_class'][indi].append(acc_partial * 100)
        
    acc['average'] = acc['average'] / n * 100
    acc['per_class_average'] = acc['per_class_average'] / len(label_unique) * 100
    
    return acc
