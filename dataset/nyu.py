#coding:utf-8

import os
import torch
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat
import numpy as np
import glob
from torchvision import transforms
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class NYUv2(data.Dataset):
    """NYUv2 depth dataset loader.
    
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **num_classes** (string, optional): The number of classes, must be 40 or 13. Default:13.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A list of function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with labels or not. Default: 'labeled'.
    """
    cmap = colormap()

    def __init__(self,
                 root,
                 split='train',
                 num_classes=13,
                 transform=None,
                 ds_type='labeled'):

        assert(split in ('train', 'test'))
        assert(ds_type in ('labeled', 'unlabeled'))
        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transform = transform
        self.num_classes = num_classes
        self.train_idx = np.array([255, ] + list(range(num_classes)))

        if ds_type == 'labeled':
            split_mat = loadmat(os.path.join(
                self.root, 'nyuv2-meta-data', 'splits.mat'))

            idxs = split_mat[self.split+'Ndxs'].reshape(-1)

            self.images = [os.path.join(self.root, self.split, "image", '%d.npy' % (idx-1))
                           for idx in idxs]
            if self.num_classes == 13:
                self.targets = [os.path.join(self.root, 'nyuv2-meta-data', '%s_labels_13' % self.split, 'new_nyu_class13_%04d.png' % idx)
                                for idx in idxs]
            elif self.num_classes == 40:
                self.targets = [os.path.join(self.root, '480_640', 'SEGMENTATION', '%04d.png' % idx)
                                for idx in idxs]
            else:
                raise ValueError(
                    'Invalid number of classes! Please use 13 or 40')
        else:
            self.images = [glob.glob(os.path.join(
                self.root, 'unlabeled_images/*.png'))]
        print(self.split, len(self.images))


    def __getitem__(self, idx):
        if self.ds_type == 'labeled':

            image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
            semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))


            image_array = np.load(self.images[idx])

            # 转换 NumPy 数组为 PIL 图像
            # 假设 image_array 的形状为 (height, width, channels)
            # 如果是灰度图像，形状应为 (height, width)
            image = Image.fromarray(image_array.astype('float32'))
            # image = Image.open(self.images[idx])
            target = Image.open(self.targets[idx])

            if self.transform:
                image, target = self.transform(image, target)
            #print(target)
            target = self.train_idx[target]
            return image, target
        else:
            image = Image.open(self.images[idx])
            if self.transforms is not None:
                image = self.transforms(image)
            image = transforms.ToTensor()(image)
            return image, None

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        target = (target+1).astype('uint8')  # 255 -> 0, 0->1, 1->2
        return cls.cmap[target]

class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_


import fnmatch


class NYUv2_M(data.Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """
    cmap = colormap()
    def __init__(self, root, mode='train', augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        if self.mode == 'train':
            data_len = len(fnmatch.filter(os.listdir(self.root + '/train/image'), '*.npy'))
            # self.index_list = data_len
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/train'
        else:
            data_len = len(fnmatch.filter(os.listdir(self.root + '/val/image'), '*.npy'))
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/val'

    def __getitem__(self, i):
        index = self.index_list[i]
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        # 调整 image 的大小
        image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear',
                                      align_corners=False).squeeze(0)

        # 调整 target 的大小
        # 注意：'nearest' 插值方法适用于离散标签
        semantic = F.interpolate(semantic.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest').squeeze(
            0).squeeze(0)


        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = - normal[0, :, :]

        # return image.float(), {'segmentation': semantic.float(), 'depth': depth.float(), 'normal': normal.float()}
        return image.float(), semantic.to(torch.uint8)
    def __len__(self):
        return len(self.index_list)

    @classmethod
    def decode_target(cls, target):
        target = (target + 1).astype('uint8')  # 255 -> 0, 0->1, 1->2
        return cls.cmap[target]


class NYUv2Depth(data.Dataset):
    """NYUv2 depth dataset loader.
    
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **num_classes** (string, optional): The number of classes, must be 40 or 13. Default:13.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A list of function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with labels or not. Default: 'labeled'.
    """
    cmap = colormap()

    def __init__(self,
                 root,
                 split='train',
                 num_classes=13,
                 transform=None,
                 #target_transforms=None,
                 ds_type='labeled'):

        assert(split in ('train', 'test'))
        assert(ds_type in ('labeled', 'unlabeled'))

        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transform = transform

        self.num_classes = num_classes

        self.train_idx = np.array([255, ] + list(range(num_classes)))
        
        if ds_type == 'labeled':
            split_mat = loadmat(os.path.join(
                self.root, 'nyuv2-meta-data', 'splits.mat'))

            idxs = split_mat[self.split+'Ndxs'].reshape(-1)
            self.images = [os.path.join(self.root, '480_640', 'IMAGE', '%d.png' % (idx-1))
                           for idx in idxs]
            if self.num_classes == 13:
                self.targets = [os.path.join(self.root, 'nyuv2-meta-data', '%s_labels_13' % self.split, 'new_nyu_class13_%04d.png' % idx)
                                for idx in idxs]
            elif self.num_classes == 40:
                self.targets = [os.path.join(self.root, '480_640', 'SEGMENTATION', '%04d.png' % idx)
                                for idx in idxs]
            else:
                raise ValueError(
                    'Invalid number of classes! Please use 13 or 40')
            self.depths = [os.path.join(
                self.root, 'FINAL_480_640', 'DEPTH', '%04d.png' % idx) for idx in idxs]
        else:
            self.images = [glob.glob(os.path.join(
                self.root, 'unlabeled_images/*.png'))]

    def __getitem__(self, idx):
        if self.ds_type == 'labeled':
            image = Image.open(self.images[idx])
            depth = Image.open(self.depths[idx])
            #print(np.array(depth,dtype='float').max())
            if self.transform:
                image, depth = self.transform(image, depth)
            return image, depth / 1000
        else:
            image = Image.open(self.images[idx])
            if self.transform is not None:
                image = self.transform(image)
            #image = transforms.ToTensor()(image)
            return image, None

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        cm = plt.get_cmap('jet')
        target = (target/7).clip(0,1)
        target = cm(target)[:,:,:,:3]
        return target
        
