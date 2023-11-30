# https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452

import torch
from torch.utils.data import BatchSampler


def get_weighted_random_sampler(dataset, label_key: str = "labels"):
    """
    Returns a WeightedRandomSampler for a given dataset.
    The weights are calculated by taking the inverse of the class frequency.
    """
    labels, class_sample_count = dataset.get_label_distribution(label_key=label_key)
    weight = 1.0 / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in labels])
    return torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
