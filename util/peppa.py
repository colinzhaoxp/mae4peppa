import os

import PIL
import pandas as pd

from torch.utils.data import Dataset

from .datasets import build_transform


class Peppa():
    def __init__(self, root):
        self.root = root
        self.train_dir = os.path.join(root, 'train')
        self.val_dir = os.path.join(root, 'val')
        self.test_dir = os.path.join(root, 'test')

        self.labels = self.make_labels(self.root)
        self.train_samples = self.make_dataset(self.train_dir)
        self.val_samples = self.make_dataset(self.val_dir)
        self.test_samples = self.make_dataset(self.test_dir)

    def make_labels(self, path):
        # labels = {(filename: weight), ...}
        path = os.path.join(path, 'label.txt')
        table = pd.read_table(path, sep=',', header=None)
        table[0] = table[0].apply(lambda x: os.path.basename(x))
        labels = table.set_index(0).to_dict()[1]
        return labels

    def make_dataset(self, dir):
        # images = [(origin_path, mask_path, weight), ...]
        mask_dir = dir.split('/')[-1] + '_mask'
        origin_dir = dir.split('/')[-1] + '_real_expend'
        miss_files = []
        images = []
        for root, _, fpaths in os.walk(os.path.join(dir, origin_dir)):
            for fpath in fpaths:
                origin_path = os.path.join(dir, origin_dir, fpath)
                mask_path = os.path.join(dir, mask_dir, fpath)
                fname = os.path.basename(origin_path).split('_')[-1]
                try:
                    weight = self.labels[fname]
                except KeyError:
                    miss_files.append(origin_path)
                    continue
                images.append((origin_path, mask_path, weight))

        if len(miss_files) > 0:
            print(f"miss files in lable.txt: {len(miss_files)}")
            print(miss_files)

        return images


class PreProcessor(Dataset):
    def __init__(self, dataset, transform=None, eval_transform=None):
        super(PreProcessor, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.eval_transform = eval_transform
        self.loader = self.default_loader

    def default_loader(self, path):
        return PIL.Image.open(path).convert('RGB')

    def __getitem__(self, index):
        path, mask_path, target = self.dataset[index]
        sample = self.loader(path)
        sample_masked = self.loader(mask_path)

        if self.transform is not None:
            sample_masked = self.transform(sample_masked)

        if self.eval_transform is not None:
            sample = self.eval_transform(sample)

        return sample_masked, sample

    def __len__(self):
        return len(self.dataset)


def build_peppa_dataset(args, transform=None):
    if transform is None:
        transform = build_transform(False, args)
    val_transform = build_transform(False, args)

    dataset = Peppa(args.data_path)

    train_dataset = PreProcessor(dataset.train_samples, transform=transform, eval_transform=val_transform)
    val_dataset = PreProcessor(dataset.val_samples, transform=val_transform)
    test_dataset = PreProcessor(dataset.test_samples, transform=val_transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    root = "/home/zhaoxp/workspace/mae-test/data/peppa"
    dataset = Peppa(root)
    a = 1
