import os

import PIL
import pandas as pd

from torch.utils.data import Dataset

from util.datasets import build_transform


class Peppa2depthV2():
    def __init__(self, root):
        self.root = root

        self.labels_train_path = os.path.join(root, 'train_random.txt')
        self.labels_val_path = os.path.join(root, 'val_random.txt')
        self.labels_test_path = os.path.join(root, 'test_random.txt')

        self.labels_train = self.make_labels(self.labels_train_path)
        self.labels_val = self.make_labels(self.labels_val_path)
        self.labels_test = self.make_labels(self.labels_test_path)

        self.rgb_path = os.path.join(root, 'rgb_75_110')
        self.depth_path = os.path.join(root, 'depth_75_110')

        self.train_samples = self.make_dataset(self.rgb_path, self.depth_path, self.labels_train)
        self.val_samples = self.make_dataset(self.rgb_path, self.depth_path, self.labels_val)
        self.test_samples = self.make_dataset(self.rgb_path, self.depth_path, self.labels_test)

        del self.labels_train, self.labels_val, self.labels_test

    def make_labels(self, file_path):
        # rgb_depth_map = {rgb_file_name: (depth_file_name, weight, posture), ...}
        table = pd.read_table(file_path, sep=',', header=None, names=['rgb_path', 'depth_path', 'weight', 'posture'])
        table['rgb_path'] = table['rgb_path'].apply(lambda x: x.split('\\')[-1])
        table['depth_path'] = table['depth_path'].apply(lambda x: x.split('/')[-1])
        rgb_depth_map = table.set_index('rgb_path').to_dict(orient='index')
        return rgb_depth_map

    def make_dataset(self, rgb_dir, depth_dir, labels):
        # images = [(origin_rgb_path, mask_rgb_path, origin_depth_path, mask_depth_path, weight), ...]
        miss_files = []
        images = []

        for rgb_file_name, values in labels.items():
            origin_rgb_path = os.path.join(rgb_dir, rgb_file_name)
            depth_file_name, weight, _ = values.values()
            origin_depth_path = os.path.join(depth_dir, depth_file_name)
            images.append((origin_rgb_path, origin_depth_path, weight))

        return images


class PreProcessor2depthV2(Dataset):
    def __init__(self, dataset, transform=None, eval_transform=None):
        super(PreProcessor2depthV2, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.eval_transform = eval_transform
        self.loader = self.default_loader

    def default_loader(self, path):
        return PIL.Image.open(path).convert('RGB')

    def __getitem__(self, index):
        origin_rgb_path, origin_depth_path, weight = self.dataset[index]

        rgb_img = self.loader(origin_rgb_path)

        # depth_img = self.loader(origin_depth_path).convert('L') # convert to gray image

        # rgbd_img = PIL.Image.merge('RGBA', (*rgb_img.split(), depth_img))
        rgbd_img = rgb_img

        if self.transform is not None:
            rgbd_img = self.transform(rgbd_img)

        if self.eval_transform is not None:
            rgbd_img = self.eval_transform(rgbd_img)

        return rgbd_img, weight

    def __len__(self):
        return len(self.dataset)


def build_peppa2depthV2_dataset(args, transform=None):
    use_depth = None
    if args.in_chans == 4:
        use_depth = (0.005261, 0.011198)

    if transform is None:
        transform = build_transform(False, use_depth, args)
    val_transform = build_transform(False, use_depth, args)

    dataset = Peppa2depthV2(args.data_path)

    train_dataset = PreProcessor2depthV2(dataset.train_samples, transform=transform)
    val_dataset = PreProcessor2depthV2(dataset.val_samples, eval_transform=val_transform)
    test_dataset = PreProcessor2depthV2(dataset.test_samples, eval_transform=val_transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    root = "/home/zhaoxp/workspace/mae-test/data/peppa2depth"
    dataset = Peppa2depthV2(root)
    a = 1
