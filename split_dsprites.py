import numpy as np
import sys
import os

if __name__ == '__main__':
    directory = sys.argv[1]
    filename = sys.argv[2]
    K = int(sys.argv[3])
    path = os.path.join(directory, filename)
    zipfile = np.load(path, allow_pickle=True, encoding='bytes')
    imgs = zipfile['imgs']
    latents_values = zipfile['latents_values']
    latents_classes = zipfile['latents_classes']
    metadata = zipfile['metadata']

    idxs_square = (latents_classes[:, 1] == 0) * (latents_classes[:, 3] < 10)
    idxs_oval = (latents_classes[:, 1] == 1) * (latents_classes[:, 3] < 10)
    idxs_heart = (latents_classes[:, 1] == 2) * (latents_classes[:, 3] < 10)

    idxs = idxs_square + idxs_oval + idxs_heart
    idxs = list(map(bool, idxs))
    print(min(idxs), max(idxs), sum(idxs), len(idxs))
    imgs = imgs[idxs]
    latents_values = latents_values[idxs]
    latents_classes = latents_classes[idxs]

    train_idxs  = (latents_classes[:, 1] == 0) * (latents_classes[:, 4] <  16) * (latents_classes[:, 5] < 16)
    train_idxs += (latents_classes[:, 1] == 1) * (latents_classes[:, 4] <  16) * (latents_classes[:, 5] >= 16)
    train_idxs += (latents_classes[:, 1] == 2) * (latents_classes[:, 4] >= 16) * (latents_classes[:, 5] < 16)

    if K == 2:
        train_idxs += (latents_classes[:, 1] == 0) * (latents_classes[:, 4] < 16) * (latents_classes[:, 5] >= 16)
        train_idxs += (latents_classes[:, 1] == 1) * (latents_classes[:, 4] <  16) * (latents_classes[:, 5] <  16)
        train_idxs += (latents_classes[:, 1] == 2) * (latents_classes[:, 4] >= 16) * (latents_classes[:, 5] >= 16)
    train_idxs = list(np.arange(len(imgs))[train_idxs])

    valid_idxs  = (latents_classes[:, 1] == 0) * (latents_classes[:, 4] >= 16) * (latents_classes[:, 5] <  16)
    valid_idxs += (latents_classes[:, 1] == 1) * (latents_classes[:, 4] >= 16) * (latents_classes[:, 5] >= 16)
    valid_idxs += (latents_classes[:, 1] == 2) * (latents_classes[:, 4] <  16) * (latents_classes[:, 5] >= 16)
    valid_idxs = list(np.arange(len(imgs))[valid_idxs])

    test_idxs = list(set(range(len(imgs))) - set(valid_idxs) - set(train_idxs))
    print(len(imgs), len(train_idxs), len(valid_idxs), len(test_idxs))

    train_imgs = np.take(imgs, train_idxs, axis=0)
    train_latents_classes = np.take(latents_classes, train_idxs, axis=0)
    train_latents_values = np.take(latents_values, train_idxs, axis=0)
    valid_imgs = np.take(imgs, valid_idxs, axis=0)
    valid_latents_classes = np.take(latents_classes, valid_idxs, axis=0)
    valid_latents_values = np.take(latents_values, valid_idxs, axis=0)
    test_imgs = np.take(imgs, test_idxs, axis=0)
    test_latents_classes = np.take(latents_classes, test_idxs, axis=0)
    test_latents_values = np.take(latents_values, test_idxs, axis=0)

    train_zipfile = {'imgs': train_imgs,
                     'latents_classes': train_latents_classes,
                     'latents_values': train_latents_values,
                     'metadata': metadata}
    np.savez_compressed(os.path.join(directory, f'dsprites_composition_train_{K}.npz'), **train_zipfile)
    valid_zipfile = {'imgs': valid_imgs,
                     'latents_classes': valid_latents_classes,
                     'latents_values': valid_latents_values,
                     'metadata': metadata}
    np.savez_compressed(os.path.join(directory, f'dsprites_composition_valid_{K}.npz'), **valid_zipfile)
    test_zipfile = {'imgs': test_imgs,
                    'latents_classes': test_latents_classes,
                    'latents_values': test_latents_values,
                    'metadata': metadata}
    np.savez_compressed(os.path.join(directory, f'dsprites_composition_test_{K}.npz'), **test_zipfile)

