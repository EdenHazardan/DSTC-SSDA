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


def main():
    all_file = '/home/gaoy/SSDA/data/city_list/train.txt'

    sup_file = '/home/gaoy/SSDA/data/city_list/train_sup_100.txt'
    unsup_file = '/home/gaoy/SSDA/data/city_list/train_unsup_100.txt'

    with open(all_file, 'r') as f:
        all_lines = f.readlines()
    # print(sup_lines)
    random.shuffle(all_lines)
    sup_lines = all_lines[:100]
    unsup_lines = all_lines[100:]
    print(len(sup_lines))
    print(len(unsup_lines))


    with open(sup_file, 'w') as f:
        for line in sup_lines:
            f.write('{}'.format(line))

    with open(unsup_file, 'w') as f:
        for line in unsup_lines:
            f.write('{}'.format(line))


if __name__ == '__main__':
    main()
