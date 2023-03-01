import os
import sys
import cv2
import math
import json
import random
import shutil
import numpy as np
import os.path as osp
# from scipy.ndimage import filters
import scipy.misc as m
import imageio 
from skimage.transform import resize
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as ttransforms
from utils.transform import crop, hflip, normalize, blur, cutout

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
CITY_MEAN = np.array((73.15835921, 82.90891754, 72.39239876), dtype=np.float32)

label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
NUM_CLASS = 19

# Labels
ignore_label = -1
cityscapes_id_to_trainid = {
    -1: ignore_label,
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,
    7: 0,
    8: 1,
    9: ignore_label,
    10: ignore_label,
    11: 2,
    12: 3,
    13: 4,
    14: ignore_label,
    15: ignore_label,
    16: ignore_label,
    17: 5,
    18: ignore_label,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: ignore_label,
    30: ignore_label,
    31: 16,
    32: 17,
    33: 18
}

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

class cityscapes_dataset(Dataset):
    def __init__(self, split='train', semi_sup=None, semi_unsup=None, pseudo=None, synthia=False):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'cityscapes', 'leftImg8bit', split)
        self.gt_path = os.path.join(self.data_path, 'cityscapes', 'gtFine', split)
        self.split = split
        self.semi_sup = semi_sup
        self.semi_unsup = semi_unsup
        self.pseudo = pseudo
        self.synthia = synthia
        
        # Viper to City Label map
        self.id_to_trainid = cityscapes_id_to_trainid

        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6,
                          7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}


        self.get_list()

    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            if self.semi_sup:
                list_path = os.path.join(self.data_path, 'city_list', 'train_sup_{}.txt'.format(self.semi_sup))
                print("load list path : ",list_path)
            elif self.semi_unsup:
                list_path = os.path.join(self.data_path, 'city_list', 'train_unsup_{}.txt'.format(self.semi_unsup))
                print("load list path : ",list_path)
            else:
                list_path = os.path.join(self.data_path, 'city_list', 'train.txt')
                print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'city_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-6:])
            filename = line.split("_")[1]
            
            if self.split == 'train':
                frame_prefix = line[6:-7]
            else:
                frame_prefix = line[4:-7]

            name = '{}/{}_{:06d}'.format(filename, frame_prefix, frame_id)
            self.im_name.append(name)

    def __len__(self):
        return len(self.gt_name)

    def img_transform(self, image):
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1024, 512), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.synthia:
            label_copy_16 = ignore_label * \
                np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        return label_copy


    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, '{}_leftImg8bit.png'.format(im_name)))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, '{}_gtFine_labelIds.png'.format(im_name)))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.pseudo:
            return img, gt, im_name
        return img, gt

class cityscapes_dataset_crop(Dataset):
    def __init__(self, split='train', semi_sup=None, semi_unsup=None, pseudo=None, synthia=False):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'cityscapes', 'leftImg8bit', split)
        self.gt_path = os.path.join(self.data_path, 'cityscapes', 'gtFine', split)
        self.split = split
        self.semi_sup = semi_sup
        self.semi_unsup = semi_unsup
        self.pseudo = pseudo
        self.crop_size = (512, 512)
        
        self.synthia = synthia
        
        # Viper to City Label map
        self.id_to_trainid = cityscapes_id_to_trainid

        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6,
                          7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}
    
        self.get_list()

    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            if self.semi_sup:
                list_path = os.path.join(self.data_path, 'city_list', 'train_sup_{}.txt'.format(self.semi_sup))
                print("load list path : ",list_path)
            elif self.semi_unsup:
                list_path = os.path.join(self.data_path, 'city_list', 'train_unsup_{}.txt'.format(self.semi_unsup))
                print("load list path : ",list_path)
            else:
                list_path = os.path.join(self.data_path, 'city_list', 'train.txt')
                print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'city_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-6:])
            filename = line.split("_")[1]
            
            if self.split == 'train':
                frame_prefix = line[6:-7]
            else:
                frame_prefix = line[4:-7]

            name = '{}/{}_{:06d}'.format(filename, frame_prefix, frame_id)
            self.im_name.append(name)

    def __len__(self):
        return len(self.gt_name)

    def img_transform(self, image):
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1024, 512), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.synthia:
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
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, '{}_leftImg8bit.png'.format(im_name)))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, '{}_gtFine_labelIds.png'.format(im_name)))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        # if self.split == 'train':
        img, gt = self.random_crop(img, gt)

        return img, gt


if __name__ == '__main__':
    dataset = cityscapes_dataset_crop(split='train', synthia=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        img, gt = data
        
        print('{}/{}'.format(i, len(dataloader)), img.shape, gt.shape, gt.max(),  gt.min())