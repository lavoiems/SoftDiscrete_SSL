import torch
from torch import nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from common.loaders import calc_stats, MultiViewDataset


class MPI3DDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        label = list(self.labels[idx] / (np.array([2, 2, 2, 2, 2, 40, 40]) - 1))
        if self.transform:
            sample = self.transform(sample)
        label[0] = int(label[0])
        label[1] = int(label[1])
        label[2] = int(label[2])
        label[3] = int(label[3])
        label[4] = int(label[4])
        return tuple([sample, ] + label)


def load(data_path, train_batch_size, test_batch_size, dataset, split='real', augment='color',
         datasplit='composition', dataset_K=1, device=None, n_train=150000, n_sub_train=150000, **kwargs):
    train_file_path = os.path.join(data_path, f'train_{datasplit}_{dataset}_{split}_{dataset_K}.npz')
    valid_file_path = os.path.join(data_path, f'valid_{datasplit}_{dataset}_{split}_{dataset_K}.npz')
    test_file_path = os.path.join(data_path, f'test_{datasplit}_{dataset}_{split}_{dataset_K}.npz')

    train = np.load(train_file_path)
    train_images = train['imgs']
    train_labels = train['latents_classes']
    valid = np.load(valid_file_path)
    valid_images = valid['imgs']
    valid_labels = valid['latents_classes']
    test = np.load(test_file_path)
    test_images = test['imgs']
    test_labels = test['latents_classes']

    if augment == 'none':
        augmentation = nn.Sequential()
    elif augment == 'crop':
        augmentation = nn.Sequential(
            transforms.RandomApply(nn.ModuleList([transforms.RandomResizedCrop((64, 64),
                                                                               scale=(0.5, 1.),
                                                                               interpolation=transforms.InterpolationMode.NEAREST), ]), p=0.9),
        )
    else:
        augmentation = nn.Sequential(
            transforms.RandomApply(nn.ModuleList([transforms.RandomResizedCrop((64, 64),
                                                                               scale=(0.5, 1.),
                                                                               interpolation=transforms.InterpolationMode.NEAREST), ]), p=0.9),
            transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(0.4, 0.4, 0.4, 0.), ]), p=0.8),
            transforms.RandomGrayscale(p=0.2),
            )
    augmentation = torch.jit.script(augmentation.to(device))

    train_full_dataset = MPI3DDataset(train_images, train_labels, transform=transforms.ToTensor())

    lengths = [n_train, len(train_full_dataset) - n_train]
    train_dataset, rest_iid_dataset = torch.utils.data.random_split(train_full_dataset, lengths,
                                                                    generator=torch.Generator().manual_seed(100))

    n_sub = min(n_sub_train, len(train_dataset))
    lengths = [n_sub, len(train_dataset) - n_sub]
    train_sub_dataset, _ = torch.utils.data.random_split(train_dataset, lengths,
                                                        generator=torch.Generator().manual_seed(101))

    train_ssl_dataset = MultiViewDataset(train_dataset, augmentation)

    n_valid_iid = min(len(rest_iid_dataset)//2, 50000)
    lengths = [n_valid_iid, len(rest_iid_dataset) - n_valid_iid]
    valid_iid_dataset, _ = torch.utils.data.random_split(rest_iid_dataset, lengths,
                                                         generator=torch.Generator().manual_seed(102))
    valid_dataset = MPI3DDataset(valid_images, valid_labels, transform=transforms.ToTensor())
    test_dataset = MPI3DDataset(test_images, test_labels, transform=transforms.ToTensor())

    train_ssl = DataLoader(train_ssl_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False,
                           pin_memory=False, num_workers=4, persistent_workers=False)
    train_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=True, drop_last=False,
                              pin_memory=False, num_workers=2, persistent_workers=False)
    sub_loader = DataLoader(train_sub_dataset, batch_size=test_batch_size, shuffle=True, drop_last=False,
                            pin_memory=False, num_workers=2, persistent_workers=False)
    valid_iid_loader = DataLoader(valid_iid_dataset, batch_size=test_batch_size, shuffle=True, drop_last=False,
                                 pin_memory=False, num_workers=2, persistent_workers=False)
    val_loader = DataLoader(valid_dataset, batch_size=test_batch_size, drop_last=False, num_workers=2,
                            persistent_workers=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=False, num_workers=2,
                             persistent_workers=False, shuffle=False)

    attribute_info = {
        0: (6, 'classification'),  # object color (6)
        1: (6, 'classification'),  # object shape (6)
        2: (2, 'classification'),  # object size (2)
        3: (3, 'classification'),  # camera height (3)
        4: (3, 'classification'),  # background color (3)
        5: (1, 'regression'),  # horizontal axis (40)
        6: (1, 'regression'),  # vertical axis (40)
        }
    norm_info = calc_stats(train_loader, device=device)
    channels = 3

    loaders = dict(
        train_ssl=train_ssl,
        train=train_loader,
        sub=sub_loader,
        iid=valid_iid_loader,
        val=val_loader,
        ood=test_loader,
    )
    return loaders, attribute_info, norm_info, channels, augmentation

