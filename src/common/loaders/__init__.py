import torch
from torch.utils.data import Dataset


def calc_stats(loader, device=None):
    N = len(loader.dataset)

    def calc_batch_mean(x):
        x = x[0].to(device)
        return x.mean((2, 3), True).double().div(N).sum(0, True).float()

    mean = sum(map(calc_batch_mean, iter(loader)))

    def calc_batch_var(x):
        x = x[0].to(device)
        return x.sub(mean).pow(2).mean((2, 3), True).double().div(N).sum(0, True).float()

    std = torch.sqrt(sum(map(calc_batch_var, iter(loader))))
    return mean, std


class MultiViewDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, *_ = self.dataset[idx]
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2
