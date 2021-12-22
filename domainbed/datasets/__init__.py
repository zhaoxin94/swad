import numpy as np
import random
from math import sqrt

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms as T

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT


def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist', 'fda']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": DBT.aug}
    elif data_type == "test":
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    elif data_type == "fda":
        # No augmentation before FDA operation
        dset.transforms = {"x": DBT.basic, "x_aug": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"
            if args.algorithm == 'FDA':
                in_type = "fda"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    dataset_all = None
    if args.algorithm == "FDA":
        dataset_all = DatasetAll_FDA([
            env for i, (env, _) in enumerate(in_splits) if i not in test_envs
        ])

    return dataset, dataset_all, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


class DatasetAll_FDA(Dataset):
    """
    Combine Seperated Datasets
    """
    def __init__(self, data_list, alpha=1.0):

        self.data = ConcatDataset(data_list)

        self.pre_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.3),
            T.RandomGrayscale(), lambda x: np.asarray(x)
        ])
        self.post_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img, label = self.data[idx]
        ret = self.data[idx]

        img = ret['x_aug']
        label = ret['y']

        # randomly sample an item from the dataset
        img_s, _ = self._sample_item()

        # do pre_transform before FDA
        img = self.pre_transform(img)
        img_s = self.pre_transform(img_s)

        # FDA
        img_mix = self._colorful_spectrum_mix(img, img_s, self.alpha)

        # do post_transform after FDA
        img = self.post_transform(img)
        img_mix = self.post_transform(img_mix)

        img = [img, img_mix]
        label = [label, label]

        ret["x"] = img
        ret["y"] = label

        del ret["x_aug"]

        return ret

    def _colorful_spectrum_mix(self, img1, img2, alpha, ratio=1.0):
        """Input image size: ndarray of [H, W, C]"""
        lam = np.random.uniform(0, alpha)

        assert img1.shape == img2.shape
        h, w, c = img1.shape
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img1_fft = np.fft.fft2(img1, axes=(0, 1))
        img2_fft = np.fft.fft2(img2, axes=(0, 1))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                            h_start:h_start + h_crop,
                                                                                            w_start:w_start + w_crop]

        img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

        img21 = img1_abs * (np.e**(1j * img1_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))

        img21 = np.uint8(np.clip(img21, 0, 255))

        return img21

    def _sample_item(self):
        idxs = list(range(len(self.data)))
        selected_idx = random.sample(idxs, 1)[0]
        ret = self.data[selected_idx]
        return ret['x_aug'], ret['y']
