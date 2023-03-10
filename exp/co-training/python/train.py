import os
import sys
import time
import numpy as np
import random
import argparse
import ast
from tqdm import trange
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as ttransforms

from tensorboardX import SummaryWriter

from data.gta5 import gta5_dataset, gta5_dataset_crop, IterLoader
from data.cityscapes import cityscapes_dataset, cityscapes_dataset_crop, IterLoader
from model.img_model import deeplabv2_img_model
from utils.metrics import runningScore
from utils.utils import dynamic_copy_paste
from utils.augmentations import get_augmentation, augment

ignore_index = -1
label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
NUM_CLASS = 19


def get_arguments():
    parser = argparse.ArgumentParser(description="Train SCNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_res101", type=str, help="path to resnet18 pretrained weight")
    parser.add_argument("--al_SDA", action="store_true", help="use al strong data augmantation or not.")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--source_batch_size", type=int)
    parser.add_argument("--target_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train_num_workers", type=int)
    parser.add_argument("--test_num_workers", type=int)
    parser.add_argument("--train_iterations", type=int)
    parser.add_argument("--early_stop", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--work_dirs", type=str)

    return parser.parse_args()

def train():

    args = get_arguments()
    print(args)
    if not os.path.exists(os.path.join(args.work_dirs, args.exp_name)):
        os.makedirs(os.path.join(args.work_dirs, args.exp_name))
    tblogger = SummaryWriter(os.path.join(args.work_dirs, args.exp_name))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    fh = logging.FileHandler(os.path.join(args.work_dirs, args.exp_name, '{}.log'.format(rq)), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('random seed:{}'.format(random_seed))

    # network
    net_1 = deeplabv2_img_model(weight_res101=args.weight_res101).cuda()
    src_weight = torch.load('/home/gaoy/DSTC-SSDA/work_dirs/warm-up/source_only/best_city.pth')
    net_1.load_state_dict(src_weight, strict=True)

    net_2 = deeplabv2_img_model(weight_res101=args.weight_res101).cuda()
    trg_weight = torch.load('/home/gaoy/DSTC-SSDA/work_dirs/warm-up/target_100_only/best_city.pth')
    net_2.load_state_dict(trg_weight, strict=True)


    # optimizer
    params_1 = net_1.segnet.optim_parameters(lr=args.lr)
    optimizer_1 = optim.SGD(params=params_1, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    params_2 = net_2.segnet.optim_parameters(lr=args.lr)
    optimizer_2 = optim.SGD(params=params_2, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    # dataset
    source_data = gta5_dataset_crop(split='train')
    source_loader = DataLoader(
        source_data,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    source_loader = IterLoader(source_loader)

    label_data = cityscapes_dataset_crop(split='train', semi_sup=100)
    label_loader = DataLoader(
        label_data,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    label_loader = IterLoader(label_loader)

    unlabel_data = cityscapes_dataset_crop(split='train', semi_unsup=100)
    unlabel_loader = DataLoader(
        unlabel_data,
        batch_size=args.target_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    unlabel_loader = IterLoader(unlabel_loader)

    print("source data:", len(source_loader))
    print("label data:", len(label_loader))
    print("unlabeled data:", len(unlabel_loader))

    test1_data = cityscapes_dataset(split='val')
    test1_loader = DataLoader(
        test1_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        pin_memory=True
    )

    L1loss = nn.L1Loss()
    miou_cal_1 = runningScore(n_classes=NUM_CLASS)
    miou_cal_2 = runningScore(n_classes=NUM_CLASS)
    if args.al_SDA:
        augmentations = get_augmentation()
    best_1_miou = 0.0
    best_2_miou = 0.0
    best_avg_miou = 0.0

    for step in trange(args.early_stop):
        net_1.train()
        net_2.train()

        # reset optimizers
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        # adapt LR if needed
        lr = poly_lr_scheduler(args, optimizer_1, iter=step)
        _ = poly_lr_scheduler(args, optimizer_2, iter=step)
        
        # init loss
        total_loss = 0

        # train net_1
        # source mix target
        image_s, gt_s = next(source_loader)
        image_t, gt_t = next(label_loader)
        gt_s = gt_s.cuda()
        gt_t = gt_t.cuda()
        loss, pred_src = net_1(image_s.cuda(), gt=gt_s.cuda(), return_feat=True)
        source_loss_1 = loss.mean()

        # alignment loss
        with torch.no_grad():
            pred_trg = net_2(image_s.cuda())

        pred_s_list=[]
        pred_t_list=[]
        for classid in torch.unique(gt_s):
            mask_id = (gt_s==classid).float().reshape(2,-1,gt_s.shape[1],gt_s.shape[2]).cuda()
            logit_source = (pred_src * mask_id).sum(2).sum(2).sum(0)/mask_id.sum()
            logit_source = logit_source.unsqueeze(0)

            logit_target = (pred_trg * mask_id).sum(2).sum(2).sum(0)/mask_id.sum()
            logit_target = logit_target.unsqueeze(0)
            pred_s_list.append(logit_source)
            pred_t_list.append(logit_target)

        pred_s_prototype = torch.cat(pred_s_list)
        pred_t_prototype = torch.cat(pred_t_list)
        alignment_loss = L1loss(pred_s_prototype, pred_t_prototype.detach())

        src_loss = source_loss_1 + alignment_loss * 0.1
        # source_loss.backward()
        src_loss.backward()


        loss = net_1(image_t.cuda(), gt=gt_t.cuda())
        ce_loss_1 = loss.mean()
        ce_loss_1.backward()


        # train on unlabeled target
        with torch.no_grad():
            # Run non-augmented image though model to get predictions
            image_u, _ = next(unlabel_loader)
            logits_u = net_2(image_u.cuda())
            pred = F.softmax(logits_u.detach(), dim=1)
            max_probs, pseudo_label = torch.max(pred.detach(), dim=1)

        query_class0 = []
        query_class1 = []
        num_class_gt0 = torch.unique(pseudo_label[0])
        num_class_gt0 = num_class_gt0[num_class_gt0!=ignore_index]
        len_gt0 = len(num_class_gt0)
        num_class_gt1 = torch.unique(pseudo_label[1])
        num_class_gt1 = num_class_gt1[num_class_gt1!=ignore_index]
        len_gt1 = len(num_class_gt1)

        class0= np.random.choice(num_class_gt0.cpu(), int((len_gt0+len_gt0%2)/2), replace=False)
        for k in range(len(class0)):
            query_class0.append(class0[k])
            
        class1 = np.random.choice(num_class_gt1.cpu(), int((len_gt1+len_gt1%2)/2), replace=False)
        for k in range(len(class1)):
            query_class1.append(class1[k])

        image_mix1, _, mask1 = dynamic_copy_paste(image_u[1].cuda(), pseudo_label[1].long(), image_u[0].cuda(), pseudo_label[0].long(), query_class0, return_mask=True) 
        image_mix2, _, mask2 = dynamic_copy_paste(image_u[0].cuda(), pseudo_label[0].long(), image_u[1].cuda(), pseudo_label[1].long(), query_class1, return_mask=True) 
        pred_mix1 = (1-mask1) * pred[1] + mask1 * pred[0]
        pred_mix2 = (1-mask2) * pred[0] + mask2 * pred[1]

        # strong augmentation on unlabeled images
        image_mix = torch.cat((image_mix1, image_mix2))
        pred_mix = torch.cat((pred_mix1.unsqueeze(0), pred_mix2.unsqueeze(0)))
        max_probs_mix, label_mix = torch.max(pred_mix.detach(), dim=1)

        if args.al_SDA:
            image_mix, label_mix = augment(image_mix.cpu(), label_mix.detach().cpu(), augmentations)
            label_mix = label_mix.cuda()


        loss = net_1(image_mix.cuda(), gt=label_mix.cuda())
        classmix_1 = loss.mean()

        classmix_1.backward()

        optimizer_1.step()

        del query_class0, query_class1, image_mix1, image_mix2, image_mix, pred_mix1, pred_mix2, pred_mix, label_mix



        # train net_2
        # source mix target
        query_class0 = []
        query_class1 = []
        num_class_gt0 = torch.unique(gt_s[0])
        num_class_gt0 = num_class_gt0[num_class_gt0!=ignore_index]
        len_gt0 = len(num_class_gt0)
        num_class_gt1 = torch.unique(gt_s[1])
        num_class_gt1 = num_class_gt1[num_class_gt1!=ignore_index]
        len_gt1 = len(num_class_gt1)
        for rc_idx in range(int((len_gt0+len_gt0%2)/2)): # half class from src
                query_class0.append(np.random.choice(num_class_gt0.cpu()))
        for rc_idx in range(int((len_gt1+len_gt1%2)/2)): # half class from src
                query_class1.append(np.random.choice(num_class_gt1.cpu()))
        image_mix1, label_mix1 = dynamic_copy_paste(image_t[0].cuda(), gt_t[0].long(), image_s[0].cuda(), gt_s[0].long(), query_class0) 
        image_mix2, label_mix2 = dynamic_copy_paste(image_t[1].cuda(), gt_t[1].long(), image_s[1].cuda(), gt_s[1].long(), query_class1) 

        image_mix = torch.cat((image_mix1, image_mix2))
        label_mix = torch.cat((label_mix1, label_mix2)).long()
        loss = net_2(image_mix.cuda(), gt=label_mix.cuda())
        mix_loss_2 = loss.mean()

        mix_loss_2.backward()

        del query_class0, query_class1, image_mix1, image_mix2, image_mix, label_mix1, label_mix2, label_mix

        # train on unlabeled target
        with torch.no_grad():
            # Run non-augmented image though model to get predictions
            logits_u = net_1(image_u.cuda())
            pred = F.softmax(logits_u.detach(), dim=1)
            max_probs, pseudo_label = torch.max(pred.detach(), dim=1)

        query_class0 = []
        query_class1 = []
        num_class_gt0 = torch.unique(pseudo_label[0])
        num_class_gt0 = num_class_gt0[num_class_gt0!=ignore_index]
        len_gt0 = len(num_class_gt0)
        num_class_gt1 = torch.unique(pseudo_label[1])
        num_class_gt1 = num_class_gt1[num_class_gt1!=ignore_index]
        len_gt1 = len(num_class_gt1)

        class0= np.random.choice(num_class_gt0.cpu(), int((len_gt0+len_gt0%2)/2), replace=False)
        for k in range(len(class0)):
            query_class0.append(class0[k])
            
        class1 = np.random.choice(num_class_gt1.cpu(), int((len_gt1+len_gt1%2)/2), replace=False)
        for k in range(len(class1)):
            query_class1.append(class1[k])

        image_mix1, _, mask1 = dynamic_copy_paste(image_u[1].cuda(), pseudo_label[1].long(), image_u[0].cuda(), pseudo_label[0].long(), query_class0, return_mask=True) 
        image_mix2, _, mask2 = dynamic_copy_paste(image_u[0].cuda(), pseudo_label[0].long(), image_u[1].cuda(), pseudo_label[1].long(), query_class1, return_mask=True) 
        pred_mix1 = (1-mask1) * pred[1] + mask1 * pred[0]
        pred_mix2 = (1-mask2) * pred[0] + mask2 * pred[1]

        image_mix = torch.cat((image_mix1, image_mix2))
        pred_mix = torch.cat((pred_mix1.unsqueeze(0), pred_mix2.unsqueeze(0)))
        max_probs_mix, label_mix = torch.max(pred_mix.detach(), dim=1)

        if args.al_SDA:
            image_mix, label_mix = augment(image_mix.cpu(), label_mix.detach().cpu(), augmentations)
            label_mix = label_mix.cuda()

        loss = net_2(image_mix.cuda(), gt=label_mix.cuda())
        classmix_2 = loss.mean()

        classmix_2.backward()

        optimizer_2.step()


        total_loss = source_loss_1 + ce_loss_1 + classmix_1 + mix_loss_2 + classmix_2


        if (step + 1) % args.log_interval == 0:

            print('iter:{}/{} lr:{:.6f} source_loss_1:{:.6f} ce_loss_1:{:.6f} classmix_1:{:.6f} mix_loss_2:{:.6f} classmix_2:{:.6f} total_loss:{:.6f} '.format(step + 1, args.train_iterations, lr, source_loss_1.item(), ce_loss_1.item(), classmix_1.item(), mix_loss_2.item(), classmix_2.item(), total_loss.item()))
            logger.info('iter:{}/{} lr:{:.6f} source_loss_1:{:.6f} ce_loss_1:{:.6f} classmix_1:{:.6f} mix_loss_2:{:.6f} classmix_2:{:.6f} total_loss:{:.6f} '.format(step + 1, args.train_iterations, lr, source_loss_1.item(), ce_loss_1.item(), classmix_1.item(), mix_loss_2.item(), classmix_2.item(), total_loss.item()))
            tblogger.add_scalar('lr', lr, step + 1)
            tblogger.add_scalar('source_loss_1', source_loss_1.item(), step + 1)
            tblogger.add_scalar('ce_loss_1', ce_loss_1.item(), step + 1)
            tblogger.add_scalar('classmix_1', classmix_1.item(), step + 1)
            tblogger.add_scalar('mix_loss_2', mix_loss_2.item(), step + 1)
            tblogger.add_scalar('classmix_2', classmix_2.item(), step + 1)
            tblogger.add_scalar('total_loss', total_loss.item(), step + 1)


        if (step + 1) % args.val_interval == 0:
        # if 1:
            print('begin validation')
            logger.info('begin validation')

            net_1.eval()
            net_2.eval()

            # test city
            test_loader_iter = iter(test1_loader)
            with torch.no_grad():
                i = 0
                for data in test_loader_iter:
                    i += 1
                    if i % 50 ==0 :
                        print("test {}/{}".format(i,len(test1_loader)))
                        # break
                    image, gt = data
                    pred_1 = net_1(image.cuda())
                    pred_2 = net_2(image.cuda())
                    pred_1 = F.softmax(pred_1.detach(), dim=1)
                    pred_2 = F.softmax(pred_2.detach(), dim=1)
                    result_1 = torch.argmax(pred_1, dim=1)
                    result_2 = torch.argmax(pred_2, dim=1)
                    result_1 = result_1.squeeze().cpu().numpy()
                    result_2 = result_2.squeeze().cpu().numpy()
                    gt = gt.squeeze().cpu().numpy()
                    miou_cal_1.update(gt, result_1)
                    miou_cal_2.update(gt, result_2)
                    miou_cal_avg.update(gt, result_avg)
                miou_1, miou_class_1 = miou_cal_1.get_scores(return_class=True)
                miou_2, miou_class_2 = miou_cal_2.get_scores(return_class=True)
                miou_cal_1.reset()
                miou_cal_2.reset()

            tblogger.add_scalar('miou_total/miou_1', miou_1, step + 1)
            tblogger.add_scalar('miou_total/miou_2', miou_2, step + 1)
            for i in range(int(NUM_CLASS)):
                tblogger.add_scalar('miou_1/{}_uda_miou'.format(label_name[i]), miou_class_1[i], step + 1)
                tblogger.add_scalar('miou_2/{}_ssl_miou'.format(label_name[i]), miou_class_2[i], step + 1)
            if miou_1 > best_1_miou:
                best_1_miou = miou_1
                save_path = os.path.join(args.work_dirs, args.exp_name, 'best_1.pth')
                torch.save(net_1.state_dict(), save_path)
            if miou_2 > best_2_miou:
                best_2_miou = miou_2
                save_path = os.path.join(args.work_dirs, args.exp_name, 'best_2.pth')
                torch.save(net_2.state_dict(), save_path)
            print('step:{} current uda miou:{:.4f} best uda miou:{:.4f}'.format(step + 1, miou_1, best_1_miou))
            logger.info('step:{} current uda miou:{:.4f} best uda miou:{:.4f}'.format(step + 1, miou_1, best_1_miou))
            print('step:{} current uda miou{}'.format(step + 1, miou_class_1))
            logger.info('step:{} current uda miou{}'.format(step + 1, miou_class_1))

            print('step:{} current ssl miou:{:.4f} best ssl miou:{:.4f}'.format(step + 1, miou_2, best_2_miou))
            logger.info('step:{} current ssl miou:{:.4f} best ssl miou:{:.4f}'.format(step + 1, miou_2, best_2_miou))
            print('step:{} current ssl miou{}'.format(step + 1, miou_class_2))
            logger.info('step:{} current ssl miou{}'.format(step + 1, miou_class_2))

def poly_lr_scheduler(args, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
    init_lr = args.lr
    max_iter = args.train_iterations
    power = 0.9
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = 10 * new_lr
    return new_lr

if __name__ == '__main__':
    train()
