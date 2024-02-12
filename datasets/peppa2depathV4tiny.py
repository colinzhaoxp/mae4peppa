import os

import PIL
import pandas as pd
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from util.datasets import build_transform

class Peppa2depthV4tiny():
    def __init__(self, root):
        self.root = root

        self.labels_train_path = os.path.join(root, 'dataV4tiny.txt')

        self.labels_train = self.make_labels(self.labels_train_path)

        self.rgb_path = os.path.join(root, 'rgb')
        self.depth_path = os.path.join(root, 'depth')

        self.train_samples = self.make_dataset(self.rgb_path, self.depth_path, self.labels_train)

        del self.labels_train

    def make_labels(self, file_path):
        # rgb_depth_map = {rgb_file_name: (depth_file_name, weight, posture), ...}
        table = pd.read_table(file_path, sep=',', header=None, names=['file_path', 'weight'])
        table['weight'] = table['weight'].astype('float32')
        rgb_map = table.set_index('file_path').to_dict(orient='index')
        return rgb_map

    def make_dataset(self, rgb_dir, depth_dir, labels):
        # images = [(origin_rgb_path, mask_rgb_path, origin_depth_path, mask_depth_path, weight), ...]
        miss_files = []
        images = []

        for rgb_file_name, weight in labels.items():
            origin_rgb_path = os.path.join(rgb_dir, rgb_file_name)
            weight = weight['weight']
            origin_depth_path = os.path.join(depth_dir, rgb_file_name)
            images.append((origin_rgb_path, origin_depth_path, weight))

        return images


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class PreProcessor2depthV4tiny(Dataset):
    def __init__(self, dataset, transform=None, eval_transform=None):
        super(PreProcessor2depthV4tiny, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.eval_transform = eval_transform
        self.loader = self.default_loader

        self.norm = transforms.Normalize(mean=mean, std=std)
        self.norm_depth = transforms.Normalize(mean=0, std=0.1)

    def default_loader(self, path):
        return PIL.Image.open(path)

    def __getitem__(self, index):
        origin_rgb_path, origin_depth_path, weight = self.dataset[index]

        rgb_img = self.loader(origin_rgb_path).convert('RGB')

        depth_img = self.loader(origin_depth_path).convert('L') # convert to gray image

        if self.transform is not None:
            state = torch.get_rng_state()
            rgb_img = self.transform(rgb_img)
            torch.set_rng_state(state)
            depth_img = self.transform(depth_img)

            rgb_img = self.norm(rgb_img)
            depth_img = self.norm_depth(depth_img)

        if self.eval_transform is not None:
            state = torch.get_rng_state()
            rgb_img = self.eval_transform(rgb_img)
            torch.set_rng_state(state)
            depth_img = self.eval_transform(depth_img)

            rgb_img = self.norm(rgb_img)
            depth_img = self.norm_depth(depth_img)

        return rgb_img, depth_img, weight

    def __len__(self):
        return len(self.dataset)


def build_peppa2depthV4tiny_dataset(args, transform=None):
    use_depth = None
    if args.in_chans == 4:
        use_depth = (0.005261, 0.011198)

    if transform is None:
        transform = build_transform(False, use_depth, args)
    val_transform = build_transform(False, use_depth, args)

    dataset = Peppa2depthV4tiny(args.data_path)

    train_dataset = PreProcessor2depthV4tiny(dataset.train_samples, transform=transform)
    val_dataset = PreProcessor2depthV4tiny(dataset.train_samples, eval_transform=val_transform)
    test_dataset = PreProcessor2depthV4tiny(dataset.train_samples, eval_transform=val_transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    root = "/home/zhaoxp/workspace/mae-test/data/peppa2depthV3"
    dataset = Peppa2depthV4tiny(root)
    a = 1
