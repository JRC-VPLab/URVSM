import os
import torch
from PIL import Image
import numpy as np
import cv2 as cv

class Retinal_loader(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.name_list = []
        self.dir_list = []

        for root, dirs, files in os.walk(self.dir):
            for file in files:
                if file.startswith('.'):
                    continue
                self.dir_list.append(os.path.join(root, file))
                self.name_list.append(file.split('.')[0])

        print(len(self.dir_list), 'testing images')
        print('Dataloader ready')

    def __len__(self):
        return len(self.name_list)

    def resize_to_nearest_multiply_of_32(self, h, w):
        new_h = round(h / 32) * 32
        new_w = round(w / 32) * 32
        return new_h, new_w

    def __getitem__(self, item):
        with Image.open(self.dir_list[item]) as img:
            img = np.asarray(img) / 255

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1).repeat(3, axis=-1)
            else:
                img = np.expand_dims(img[:,:,1], axis=-1).repeat(3, axis=-1)

            h, w, _ = img.shape
            assert h == w
            n_h, n_w = self.resize_to_nearest_multiply_of_32(h, w)

            if n_h != h:
                img = cv.resize(img, (n_h, n_w), interpolation=cv.INTER_CUBIC)

            return img, self.name_list[item]


class Retinal_loader_eval(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, test_img_size=(768, 768)):
        super().__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if not f.startswith('.')])
        self.gt_list = sorted([f for f in os.listdir(self.gt_dir) if not f.startswith('.')])
        self.test_img_size = test_img_size

        assert len(self.img_list) == len(self.gt_list)
        print(len(self.img_list), 'testing images')
        print('Dataloader ready')

    def __len__(self):
        return len(self.img_list)

    def pad_to_square(self, img, h, w):
        size = max(h, w)

        pad_h = size - h
        pad_w = size - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if img.ndim == 2:
            padding = (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
            )
        elif img.ndim == 3:
            padding = (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0)
            )

        return np.pad(img, padding, mode='constant', constant_values=0)

    def __getitem__(self, item):
        with Image.open(os.path.join(self.img_dir, self.img_list[item])) as img:
            img = np.asarray(img) / 255

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1).repeat(3, axis=-1)
            else:
                img = np.expand_dims(img[:,:,1], axis=-1).repeat(3, axis=-1)

            h, w, _ = img.shape
            if h != w:
                img = self.pad_to_square(img, h, w)

            img = cv.resize(img, (self.test_img_size[0], self.test_img_size[1]), interpolation=cv.INTER_CUBIC)

        with Image.open(os.path.join(self.gt_dir, self.gt_list[item])) as gt:
            gt = np.asarray(gt) / 255
            if len(gt.shape) == 3:
                gt = gt[:,:,0]

            h, w = gt.shape
            if h != w:
                gt = self.pad_to_square(gt, h, w)

        return img, gt, self.img_list[item]