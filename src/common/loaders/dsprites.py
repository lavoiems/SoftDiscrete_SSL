import torch
from torch import nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from common.loaders import calc_stats, MultiViewDataset


class SpritesDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float().unsqueeze(1)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        # XXX Don't use ToTensor here, they have already prepared data to be in (B, C, H, W) shape
        label = list(self.labels[idx] / np.array([1, 5, 1, 31, 31]))
        label[0] = int(label[0])
        label[2] = int(label[2])
        return tuple([sample, ] + label)


def load(data_path, train_batch_size, test_batch_size, dataset, augment='crop',
         datasplit='composition', dataset_K=1, device=None, n_train=150000, n_sub_train=150000, **kwargs):
    train_file_path = os.path.join(data_path, f'dsprites_{datasplit}_train_{dataset_K}.npz')
    valid_file_path = os.path.join(data_path, f'dsprites_{datasplit}_valid_{dataset_K}.npz')
    test_file_path = os.path.join(data_path, f'dsprites_{datasplit}_test_{dataset_K}.npz')

    train_data = np.load(train_file_path)
    valid_data = np.load(valid_file_path)
    test_data = np.load(test_file_path)

    train_images = train_data['imgs']
    train_labels = train_data['latents_classes'][:, 1:]
    valid_images = valid_data['imgs']
    valid_labels = valid_data['latents_classes'][:, 1:]
    test_images = test_data['imgs']
    test_labels = test_data['latents_classes'][:, 1:]

    if augment == 'crop':
        augment = nn.Sequential(
            transforms.RandomApply(nn.ModuleList([transforms.RandomResizedCrop((64, 64),
                                                                               scale=(0.81, 1.),
                                                                               ratio=(1., 1.),
                                                                               interpolation=transforms.InterpolationMode.NEAREST), ]), p=0.9),
            transforms.RandomApply(nn.ModuleList([transforms.RandomAffine(4.,
                                                                          interpolation=transforms.InterpolationMode.NEAREST), ]), p=0.9),
            )
    elif augment == 'resize':
        augment = nn.Sequential(
            transforms.RandomAffine(4.,
                                    scale=(0.81, 1.),
                                    interpolation=transforms.InterpolationMode.NEAREST)
            )
    elif augment == 'gaussian':
        augment = nn.Sequential(
            transforms.GaussianBlur(5.)
            )
    else:
        augment = nn.Identity()

    augment = torch.jit.script(augment.to(device))

    train_full_dataset = SpritesDataset(train_images, train_labels)

    lengths = [n_train, len(train_full_dataset) - n_train]
    train_dataset, rest_iid_dataset = torch.utils.data.random_split(train_full_dataset, lengths,
                                                                    generator=torch.Generator().manual_seed(100))

    n_sub = min(n_sub_train, len(train_dataset))
    lengths = [n_sub, len(train_dataset) - n_sub]
    train_sub_dataset, _ = torch.utils.data.random_split(train_dataset, lengths,
                                                      generator=torch.Generator().manual_seed(101))

    train_ssl_dataset = MultiViewDataset(train_dataset, augment)


    n_valid_iid = min(len(rest_iid_dataset)//2, 50000)
    lengths = [n_valid_iid, len(rest_iid_dataset) - n_valid_iid]
    valid_iid_dataset, _ = torch.utils.data.random_split(rest_iid_dataset, lengths,
                                                         generator=torch.Generator().manual_seed(102))
    valid_dataset = SpritesDataset(valid_images, valid_labels)
    test_dataset = SpritesDataset(test_images, test_labels)

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
                             persistent_workers=False, shuffle=True)

    attribute_info = {
        0: (3, 'classification'),  # object shape (3)
        1: (1, 'regression'),  # object size (6)
        2: (10, 'classification'),  # orientation (40)
        3: (1, 'regression'),  # position x (32)
        4: (1, 'regression'),  # position y (32)
        }
    norm_info = calc_stats(train_loader, device=device)
    channels = 1


    loaders = dict(
        train_ssl=train_ssl,
        train=train_loader,
        sub=sub_loader,
        iid=valid_iid_loader,
        val=val_loader,
        ood=test_loader,
    )
    return loaders, attribute_info, norm_info, channels, augment

