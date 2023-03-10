import os
import sys
import cv2
import shutil
import math
import json
import random
import numpy as np
import os.path as osp
from PIL import Image, ImageFile


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as ttransforms

import imageio
os.environ['IMAGEIO_USERDIR'] = '/home/gaoy/Video_domain_adaptation_segmentation/freeimage'
imageio.core.util.appdata_dir("imageio")
imageio.plugins.freeimage.download()
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "sky", "person", "rider", "car", "bus", "motocycle", "bicycle"]
NUM_CLASS = 16

# label_name=["road", "sidewalk", "building", "light", "sign", "vegetation", "sky", "person", "rider", "car", "bus", "motocycle", "bicycle"]
# NUM_CLASS = 13

synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)



class synthia_dataset(Dataset):
    def __init__(self, split='train', pseudo=None):
        self.data_path = '/home/gaoy/DSTC-SSDA/data'
        self.im_path = os.path.join(self.data_path, 'synthia/RGB')
        self.gt_path = os.path.join(self.data_path, 'synthia/GT/LABELS')
        self.split = split
        self.pseudo = pseudo
        
        # map to cityscape's ids
        self.id_to_trainid = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13, 9: 7, 10: 11, 11: 18, 12: 17, 15: 6, 16: 9, 17: 12, 18: 14, 19: 15, 20: 16, 21: 3}

        # Only consider 16 shared classes
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}

        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'synthia_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'synthia_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        # image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        # gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        
        label_copy_16 = ignore_label * \
            np.ones(label.shape, dtype=np.float32)
        for k, v in self.trainid_to_16id.items():
            label_copy_16[label_copy == k] = v
        label_copy = label_copy_16
        return label_copy

    def __getitem__(self, idx):
        id = int(self.im_name[idx])
        im_name = f"{id:0>7d}.png"

        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        # gt = Image.open(gt_path)
        
        gt_image = imageio.imread(gt_path, format='PNG-FI')[:, :, 0]
        gt = Image.fromarray(np.uint8(gt_image))

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.pseudo:
            return img, gt, im_name
        return img, gt

class synthia_dataset_crop(Dataset):
    def __init__(self, split='train', pseudo=None):
        self.data_path = '/home/gaoy/DSTC-SSDA/data'
        self.im_path = os.path.join(self.data_path, 'synthia/RGB')
        self.gt_path = os.path.join(self.data_path, 'synthia/GT/LABELS')
        self.split = split
        self.pseudo = pseudo
        
        # map to cityscape's ids
        self.id_to_trainid = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13, 9: 7, 10: 11, 11: 18, 12: 17, 15: 6, 16: 9, 17: 12, 18: 14, 19: 15, 20: 16, 21: 3}

        # Only consider 16 shared classes
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}

        self.crop_size = (512, 512)

        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'synthia_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'synthia_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        # image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        # gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        
        label_copy_16 = ignore_label * \
            np.ones(label.shape, dtype=np.float32)
        for k, v in self.trainid_to_16id.items():
            label_copy_16[label_copy == k] = v
        label_copy = label_copy_16
        return label_copy

    def random_crop(self, img, gt):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        # start_h = random.randint(0, h - crop_h - 1)
        # start_w = random.randint(0, w - crop_w - 1)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        img = img[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
        gt = gt[start_h : start_h + crop_h, start_w : start_w + crop_w]
        return img, gt

    def __getitem__(self, idx):
        id = int(self.im_name[idx])
        im_name = f"{id:0>7d}.png"

        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        # gt = Image.open(gt_path)
        
        gt_image = imageio.imread(gt_path, format='PNG-FI')[:, :, 0]
        gt = Image.fromarray(np.uint8(gt_image))

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img, gt = self.random_crop(img, gt)

        if self.pseudo:
            return img, gt, im_name
        return img, gt

