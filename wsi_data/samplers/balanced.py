# https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
import torch
import numpy as np


def get_weighted_random_sampler(
    dataset, label_key: str = "labels", replacement: bool = True
):
    """
    Returns a WeightedRandomSampler for a given dataset.
    The weights are calculated by taking the inverse of the class frequency.
    """
    class_sample_count, labels = dataset.get_label_distribution(label_key=label_key)
    class_sample_count = class_sample_count[1]
    if isinstance(class_sample_count, list):
        class_sample_count = np.array(class_sample_count)
    weight = 1.0 / class_sample_count.astype(float)
    samples_weight = torch.tensor([weight[t] for t in labels])
    return torch.utils.data.WeightedRandomSampler(
        weights=samples_weight, num_samples=len(samples_weight), replacement=replacement
    )
