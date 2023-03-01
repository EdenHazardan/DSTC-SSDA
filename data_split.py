import json
import os
import os.path as osp
import sys
import random
from typing import overload
import torch
import numpy as np
import pickle
from PIL import Image


def main2():
    all_file = '/home/gaoy/SSDA/data/city_list/train.txt'

    sup_file = '/home/gaoy/SSDA/data/city_list/train_sup_1000_2.txt'
    unsup_file = '/home/gaoy/SSDA/data/city_list/train_unsup_1000_2.txt'

    with open(all_file, 'r') as f:
        all_lines = f.readlines()
    # print(sup_lines)
    random.shuffle(all_lines)
    sup_lines = all_lines[:1000]
    unsup_lines = all_lines[1000:]
    print(len(sup_lines))
    print(len(unsup_lines))


    with open(sup_file, 'w') as f:
        for line in sup_lines:
            f.write('{}'.format(line))

    with open(unsup_file, 'w') as f:
        for line in unsup_lines:
            f.write('{}'.format(line))


def main():
    all_file = '/data/gaoy/SSDA/work_dirs/pseudo_label/image_level/source_only_2875/aachen/aachen_000000_000019.png'
    gt_image = Image.open(all_file)
    target = np.asarray(gt_image, np.float32)
    target = torch.from_numpy(target.astype(np.int64)).long()
    print(target)
    print(target.shape)
    print(target.max())


def main3():
    # trans to DAFormer

    file_data = ""

    split = '/data/gaoy/SSDA/dataset/cityscapes/city_list/train_unsup_500_1.txt'
    with open(split, 'r') as f:
        for line in f:
            name_list = line.split("_")
            print(name_list)
            new_line = name_list[1] + '/' + name_list[1] + '_' + name_list[2] + '_' + name_list[3] 
            print(new_line)
            file_data += new_line

    with open(split, 'w') as f:
        f.write(file_data)

if __name__ == '__main__':
    # main(1488)
    # main(1488)
    # main(2975)

    # main2()
    main3()
