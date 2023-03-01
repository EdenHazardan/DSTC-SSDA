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

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
NUM_CLASS = 19
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


def get_rcs_class_probs(data_root, temperature=0.01):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
        # print("sample_class_stats = ",len(sample_class_stats))
    data_len = len(sample_class_stats)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy(), data_len


class gta5_dataset(Dataset):
    def __init__(self, split='train', pseudo=None):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.pseudo = pseudo
        
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.pseudo:
            return img, gt, im_name
        return img, gt

class gta5_dataset_intro(Dataset):
    def __init__(self, split='train', pseudo=None):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.pseudo = pseudo
        
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        list_path = os.path.join(self.data_path, 'gta5_list', 'intro.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

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
        return label_copy

    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.pseudo:
            return img, gt, im_name
        return img, gt

class gta5_dataset_crop_rcs(Dataset):
    def __init__(self, split='train'):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.crop_size = (512, 512)
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.rcs_classes, self.rcs_classprob, self.data_len = get_rcs_class_probs('/data/gaoy/SSDA/dataset/GTA5')
        
        self.rcs_class_temp = 0.01
        self.rcs_min_crop_ratio = 0.5
        self.rcs_min_pixels = 3000
        # data process
        # self.get_list()
        with open(osp.join('/data/gaoy/SSDA/dataset/GTA5', 'samples_with_class.json'), 'r') as of:
            samples_with_class_and_n = json.load(of)
        self.samples_with_class_and_n = {
            int(k): v
            for k, v in samples_with_class_and_n.items()
            if int(k) in self.rcs_classes
        }
        self.samples_with_class = {}
        for c in self.rcs_classes:
            self.samples_with_class[c] = []
            for file, pixels in self.samples_with_class_and_n[c]:
                if pixels > self.rcs_min_pixels:
                    self.samples_with_class[c].append(file)
                    # self.samples_with_class[c].append(file.split('/')[-1])
            assert len(self.samples_with_class[c]) > 0

    # def get_list(self):
    #     self.im_name = []
    #     self.gt_name = []

    #     if self.split == 'train':
    #         list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
    #         print("load list path : ",list_path)
    #     else:
    #         list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
    #     with open(list_path, 'r') as f:
    #         lines = f.readlines()

    #     for line in lines:
    #         line = line.strip()
    #         self.gt_name.append(line)
    #         self.im_name.append(line)
    
    # def get_rare_class_sample(self):
    #     c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
    #     f1 = np.random.choice(self.samples_with_class[c])
    #     print("c = ",c)
    #     print("f1 = ",f1)

    #     # i1 = self.file_to_idx[f1]
    #     # s1 = self.source[i1]
    #     if self.rcs_min_crop_ratio > 0:
    #         for j in range(10):
    #             n_class = torch.sum(s1['gt_semantic_seg'].data == c)
    #             # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
    #             if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
    #                 break
    #             # Sample a new random crop from source image i1.
    #             # Please note, that self.source.__getitem__(idx) applies the
    #             # preprocessing pipeline to the loaded image, which includes
    #             # RandomCrop, and results in a new crop of the image.
    #             # s1 = self.source[i1]
    #     return c, f1

    def __len__(self):
        return self.data_len


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
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

    def random_crop_class(self, img, gt, class_id):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        gt_t2n = gt.numpy()
        index = np.argwhere(gt_t2n==class_id)
        # print("index = ",index)
        # print("index = ",len(index))
        index = index[int(len(index)/2)]
        # print("crop_h = ",crop_h)
        # start_h = random.randint(0, h - crop_h)
        # start_w = random.randint(0, w - crop_w)
        class_h = int(index[0])
        class_w = int(index[1])
        
        start_h = random.randint(0, h - crop_h)
        
        if class_w < w/2:
            # start_w = random.randint(0, class_w)
            start_w = int(class_w/2)
        else:
            start_w = int((class_w - crop_w + w - crop_w)/2)
        img = img[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
        gt = gt[start_h : start_h + crop_h, start_w : start_w + crop_w]
        return img, gt
    
    def __getitem__(self, idx):
        class_id = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        label_path = np.random.choice(self.samples_with_class[class_id])
        # print("class_id = ",class_id)
        # print("self.samples_with_class[class_id] = ",self.samples_with_class[class_id])
        # print("label_path = ",label_path)
        im_name = label_path.split('/')[-1][0:5]+'.png'
        
        # print("im_name = ",im_name)
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img_crop, gt_crop = self.random_crop_class(img, gt, class_id)
            
            # n_class = torch.sum(gt_crop == class_id)
            # print("n_class = ",n_class)
            # for j in range(10):
            #     n_class = torch.sum(gt_crop == class_id)
            #     print("n_class = ",n_class)
            #     # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
            #     if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
            #         break
            #     # Sample a new random crop from source image i1.
            #     # Please note, that self.source.__getitem__(idx) applies the
            #     # preprocessing pipeline to the loaded image, which includes
            #     # RandomCrop, and results in a new crop of the image.
            # #     s1 = self.source[i1]
            #     img_crop, gt_crop = self.random_crop_class(img, gt, class_id)
            # img, gt = self.random_crop(img, gt)

        return img_crop, gt_crop

class gta5_dataset_crop_subset(Dataset):
    def __init__(self, split='train', class_id=None):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.class_id = class_id
        self.crop_size = (512, 512)
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.class_id == None:
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'train_with_{}.txt'.format((self.class_id)))
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    # def random_crop_class(self, img, gt):
    #     h, w = gt.shape
    #     crop_h, crop_w = self.crop_size
    #     gt_t2n = gt.numpy()
    #     index = np.argwhere(gt_t2n==self.class_id)
    #     # print("index = ",index)
    #     # print("index = ",len(index))
    #     index = index[int(len(index)/2)]
    #     # print("crop_h = ",crop_h)
    #     # start_h = random.randint(0, h - crop_h)
    #     # start_w = random.randint(0, w - crop_w)
    #     class_h = int(index[0])
    #     class_w = int(index[1])
        
    #     start_h = random.randint(0, h - crop_h)
        
    #     if class_w < w/2:
    #         start_w = random.randint(0, class_w)
    #     else:
    #         start_w = random.randint(class_w - crop_w, w - crop_w)
    #     img = img[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
    #     gt = gt[start_h : start_h + crop_h, start_w : start_w + crop_w]
    #     return img, gt
    
    def random_crop_class(self, img, gt):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        gt_t2n = gt.numpy()
        index = np.argwhere(gt_t2n==self.class_id)
        # print("index = ",index)
        # print("index = ",len(index))
        index = index[int(len(index)/2)]
        # print("crop_h = ",crop_h)
        # start_h = random.randint(0, h - crop_h)
        # start_w = random.randint(0, w - crop_w)
        class_h = int(index[0])
        class_w = int(index[1])
        
        start_h = random.randint(0, h - crop_h)
        
        if class_w < w/2:
            # start_w = random.randint(0, class_w)
            start_w = int(class_w/2)
        else:
            start_w = int((class_w - crop_w + w - crop_w)/2)
        img = img[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
        gt = gt[start_h : start_h + crop_h, start_w : start_w + crop_w]
        return img, gt
    
    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img, gt = self.random_crop_class(img, gt)

        return img, gt


class gta5_dataset_NCM(Dataset):
    def __init__(self, split='train'):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((160, 90), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        return img, gt


class gta5_dataset_v2(Dataset):
    '''
    512 * 1024
    '''
    def __init__(self, split='train'):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

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
        return label_copy

    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        return img, gt

class gta5_city_mix_dataset(Dataset):
    '''
    source and target_label
    '''
    def __init__(self, split='train', pseudo_label_path='', semi_sup=None):
        self.data_path = '/home/gaoy/SSDA/data'
        self.city_path = os.path.join(self.data_path, 'cityscapes', 'leftImg8bit', split)
        self.city_gt_path = os.path.join(self.data_path, 'cityscapes', 'gtFine', split)
        self.gta5_path = os.path.join(self.data_path, 'gta5/images')
        self.gta5_gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.semi_sup = semi_sup
        self.split = split
        
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.src_labeled_ids = []
        self.trg_labeled_ids = []

        src_list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
        print("load list path : ", src_list_path )
        if self.semi_sup:
            trg_list_path = os.path.join(self.data_path, 'city_list', 'train_sup_{}.txt'.format(self.semi_sup))
        else:
            trg_list_path = os.path.join(self.data_path, 'city_list', 'train.txt')
        print("load list path : ", trg_list_path )

        # labeled image
        with open(src_list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.src_labeled_ids.append(line)

        # unlabeled image
        with open(trg_list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            frame_id = int(line[-6:])
            filename = line.split("_")[1]
            
            if self.split == 'train':
                frame_prefix = line[6:-7]
            else:
                frame_prefix = line[4:-7]

            name = '{}/{}_{:06d}'.format(filename, frame_prefix, frame_id)
            self.trg_labeled_ids.append(name)

        self.ids = self.src_labeled_ids + self.trg_labeled_ids* math.ceil(len(self.src_labeled_ids) / len(self.trg_labeled_ids))

    def __len__(self):
        return len(self.ids)


    def img_transform(self, image):
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy


    def __getitem__(self, idx):
        id = self.ids[idx]
        
        if id in self.src_labeled_ids:
            image_path = os.path.join(os.path.join(self.gta5_path, id))
            im = Image.open(image_path).convert("RGB")
            im = im.resize((1280, 720), Image.BICUBIC)
            gt_path = os.path.join(os.path.join(self.gta5_gt_path, id))
            gt = Image.open(gt_path)
            gt = gt.resize((1280, 720), Image.NEAREST)

            # data normalization
            img = self.img_transform(im)
            gt = self._mask_transform(gt)

            return img, gt
        
        else:
            image_path = os.path.join(os.path.join(self.city_path, '{}_leftImg8bit.png'.format(id)))
            img = Image.open(image_path).convert("RGB")
            img = img.resize((1024, 512), Image.BICUBIC)
        
            gt_path = os.path.join(os.path.join(self.city_gt_path, '{}_gtFine_labelIds.png'.format(id)))
            gt = Image.open(gt_path)
            gt = gt.resize((1024, 512), Image.NEAREST)

            # data normalization
            img = self.img_transform(img)
            gt = self._mask_transform(gt)

            return img, gt


class gta5_dataset_crop(Dataset):
    def __init__(self, split='train'):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.crop_size = (512, 512)
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
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
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img, gt = self.random_crop(img, gt)

        return img, gt

class gta5_dataset_crop_analysis(Dataset):
    def __init__(self, split='train', gt_path=''):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = gt_path
        self.split = split
        self.crop_size = (512, 512)
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
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
        # label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        # for k, v in self.id_to_trainid.items():
        label[label == 255] = -1
        return label

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
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img, gt = self.random_crop(img, gt)

        return img, gt


class gta5_dataset_1(Dataset):
    def __init__(self, split='train', pseudo=None):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.pseudo = pseudo
        
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'gta5_list', '1.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'gta5_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, idx):
        im_name = self.im_name[idx]
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.pseudo:
            return img, gt, im_name
        return img, gt


class gta5_dataset_crop_train_subset(Dataset):
    def __init__(self, split='train'):
        self.data_path = '/home/gaoy/SSDA/data'
        self.im_path = os.path.join(self.data_path, 'gta5/images')
        self.gt_path = os.path.join(self.data_path, 'gta5/labels')
        self.split = split
        self.crop_size = (512, 512)
        # map to cityscape's ids
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        # if self.split == 'train':
        list_path = os.path.join(self.data_path, 'gta5_list', 'train_with_train_new.txt')
        print("load list path : ",list_path)

        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)
            self.im_name.append(line)

    def __len__(self):
        return len(self.gt_name)


    def img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image


    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target



    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
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
        image_path = os.path.join(os.path.join(self.im_path, im_name))
        im = Image.open(image_path).convert("RGB")
    
        gt_path = os.path.join(os.path.join(self.gt_path, im_name))
        gt = Image.open(gt_path)

        # data normalization
        img = self.img_transform(im)
        gt = self._mask_transform(gt)

        if self.split == 'train':
            img, gt = self.random_crop(img, gt)

        return img, gt


if __name__ == '__main__':
    # dataset = gta5_dataset(split='train', pseudo=1)
    
    # # dataset = gta5_city_mix_dataset(split='train')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # with open(os.path.join('/home/gaoy/SSDA/data/gta5_list/train_100_with_bus.txt'), 'w') as f:
    #     for i, data in enumerate(dataloader):
    #         img, gt, name = data
    #         gt_down_4x_s = F.interpolate(gt.unsqueeze(0).float(), scale_factor=0.125, mode='nearest').squeeze().cuda()
    #         if (15 in torch.unique(gt_down_4x_s)):
    #             print(name[0])
    #             f.write(name[0] + '\n')
    #         if i % 50 ==0 :
    #             print('{}/{}'.format(i, len(dataloader)), img.shape, gt.shape, gt.max(),  gt.min())

    # class_all = torch.zeros(19)
    # print("class_all = ",class_all)
    # for i, data in enumerate(dataloader):
    #     img, gt, name = data
    #     if i % 50 ==0 :
    #         print('{}/{}'.format(i, len(dataloader)))

    #     class_map = torch.unique(gt)
    #     for j in range(19):
    #         if j in class_map:
    #             class_all[j] += 1

    # for i in range(19):
    #     print("{} : {}, {}".format(label_name[i],class_all[i],class_all[i]/len(dataloader)) )
    
    
    dataset = gta5_dataset_crop_rcs()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader):
        img, gt, im_name = data
        # print("im_name = ",im_name)


    # dataset = gta5_dataset_crop_subset(split='train', class_id=16)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # for i, data in enumerate(dataloader):
    #     img, gt = data
    #     if 16 in torch.unique(gt):
    #         print("yes")
    #     else:
    #         print("no")
    # rcs_classes, rcs_classprob =  get_rcs_class_probs('/data/gaoy/SSDA/dataset/GTA5')
    # print("rcs_classes = ",rcs_classes)
    # print("rcs_classprob = ",rcs_classprob)