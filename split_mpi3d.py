import argparse
import numpy as np
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('--filename', '-f', type=str, default='./mpi3d_real.npz')
    parser.add_argument('--split', type=str, default='real')
    parser.add_argument('--K', type=int, default=4)
    args = parser.parse_args()

    zipfile = np.load(args.filename, allow_pickle=True, encoding='bytes')
    #imgs = zipfile['images']#.reshape(6, 6, 2, 3, 3, 40, 40, 64, 64, 3)
    imgs = zipfile['images'].reshape(-1, 64, 64, 3)
    attributes = np.asarray([6, 6, 2, 3, 3, 40, 40])
    a = [slice(0, attr) for attr in attributes]
    data = np.mgrid[a].reshape(len(attributes), -1).T
    data = np.ascontiguousarray(data)

    K = args.K
    assert(K >= 1)
    train_idxs = (data[:, 1] == data[:, 0])
    for j in range(1, K):
        train_idxs += (data[:, 1] + j) % 6 == data[:, 0]
    valid_idxs = (data[:, 1] + 4) % 6 == data[:, 0] # Always the same validation set. For K=5, valid is in dist. to train. For K=6, valid and test are in dist. to train
    if K <= 6:
        test_idxs = ~(train_idxs | valid_idxs)
    else:
        raise RuntimeError("Since object shape and color can take 6 distinct categorical values, we allow 0 < K < 5.")

    train_imgs = imgs[train_idxs]
    train_data = data[train_idxs]
    valid_imgs = imgs[valid_idxs]
    valid_data = data[valid_idxs]
    test_imgs = imgs[test_idxs]
    test_data = data[test_idxs]

    print(len(imgs), len(train_imgs), len(valid_imgs), len(test_imgs))

    train_zipfile = {'imgs': train_imgs,
                     'latents_classes': train_data}
    np.savez_compressed(os.path.join(args.directory, f'train_composition_mpi3d_{args.split}_{K}.npz'), **train_zipfile)
    valid_zipfile = {'imgs': valid_imgs,
                     'latents_classes': valid_data}
    np.savez_compressed(os.path.join(args.directory, f'valid_composition_mpi3d_{args.split}_{K}.npz'), **valid_zipfile)
    test_zipfile = {'imgs': test_imgs,
                    'latents_classes': test_data}
    np.savez_compressed(os.path.join(args.directory, f'test_composition_mpi3d_{args.split}_{K}.npz'), **test_zipfile)

