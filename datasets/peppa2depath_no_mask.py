import os

import PIL
import pandas as pd

from torch.utils.data import Dataset

from util.datasets import build_transform


class Peppa2depth_no_Mask():
    def __init__(self, root):
        self.root = root

        self.labels_train_path = os.path.join(root, 'train_select_seg.txt')
        self.labels_val_path = os.path.join(root, 'val_select_seg.txt')

        self.labels_train = self.make_labels(self.labels_train_path)
        self.labels_val = self.make_labels(self.labels_val_path)

        self.rgb_path = os.path.join(root, 'rgb-select')
        self.depth_path = os.path.join(root, 'depth-select')

        self.train_rgb_dir = os.path.join(root, 'rgb-occlusion-train')
        self.train_depth_dir = os.path.join(root, 'depth-occlusion-train')

        self.val_rgb_dir = os.path.join(root, 'rgb-occlusion-val')
        self.val_depth_dir = os.path.join(root, 'depth-occlusion-val')

        self.train_samples = self.make_dataset(self.train_rgb_dir, self.train_depth_dir, self.labels_train)
        self.val_samples = self.make_dataset(self.val_rgb_dir, self.val_depth_dir, self.labels_val)

        del self.labels_train, self.labels_val

    def make_labels(self, file_path):
        # rgb_depth_map = {rgb_file_name: (depth_file_name, weight, posture), ...}
        table = pd.read_table(file_path, sep=',', header=None, names=['rgb_path', 'depth_path', 'weight', 'posture'])
        table['rgb_path'] = table['rgb_path'].apply(lambda x: x.split('\\')[-1])
        table['depth_path'] = table['depth_path'].apply(lambda x: x.split('/')[-1])
        rgb_depth_map = table.set_index('rgb_path').to_dict(orient='index')
        return rgb_depth_map

    def make_dataset(self, rgb_mask_dir, depth_mask_dir, labels):
        # images = [(origin_rgb_path, mask_rgb_path, origin_depth_path, mask_depth_path, weight), ...]
        miss_files = []
        images = []
        for root, _, fpaths in os.walk(rgb_mask_dir):
            for fpath in fpaths:
                # 获得原始rgb图像路径
                rgb_file_name = fpath.split('_')[0] + '.jpg'
                origin_rgb_path = os.path.join(self.rgb_path, rgb_file_name)

                try:
                    depth_file_name, weight, _ = labels[rgb_file_name].values()
                except KeyError:
                    miss_files.append(origin_rgb_path)
                    continue

                origin_depth_path = os.path.join(self.depth_path, depth_file_name)
                images.append((origin_rgb_path, origin_depth_path, weight))

        if len(miss_files) > 0:
            print(f"miss files in lable.txt: {len(miss_files)}")
            print(miss_files)

        return images


class PreProcessor2depth_no_mask(Dataset):
    def __init__(self, dataset, transform=None, eval_transform=None):
        super(PreProcessor2depth_no_mask, self).__init__()
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
        elif self.eval_transform is not None:
            rgbd_img = self.eval_transform(rgbd_img)

        return rgbd_img, weight

    def __len__(self):
        return len(self.dataset)


def build_peppa2depth_no_mask_dataset(args, transform=None):
    use_depth = None
    if args.in_chans == 4:
        use_depth = (0.005261, 0.011198)

    if transform is None:
        transform = build_transform(False, use_depth, args)
    val_transform = build_transform(False, use_depth, args)

    dataset = Peppa2depth_no_Mask(args.data_path)

    total = len(dataset.train_samples) + len(dataset.val_samples)
    train_num = int(total * 0.8)
    val_num = int(total * 0.1)
    test_num = total - train_num - val_num

    samples = dataset.train_samples + dataset.val_samples

    train_dataset = PreProcessor2depth_no_mask(samples[:train_num], transform=transform)
    val_dataset = PreProcessor2depth_no_mask(samples[train_num: train_num+val_num], eval_transform=val_transform)
    test_dataset = PreProcessor2depth_no_mask(samples[train_num+val_num:], eval_transform=val_transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    root = "/home/zhaoxp/workspace/mae-test/data/peppa2depth"
    dataset = Peppa2depth_no_Mask(root)
    a = 1