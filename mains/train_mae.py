# -*- coding: utf-8 -*-
import os, sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn
from configs.train_options import TrainOptions
import time
import numpy as np

from models.networks2d import ResMultitaskAutoEncoder as MAE2D, init_net, GANLoss, get_scheduler
from data.ixi_loader import h5IXI, TestIXI
from data.retouch_loader import h5RETOUCH, TestRETOUCH
from data.msseg_loader import msseg, msseg_test
from data.msseg_h5 import scanner1, scanner2, scanner3, crop_ranges
from mains.test_fid import evaluate_domain_2d, evaluate_domain_3d

import logging
from tqdm import tqdm
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from tensorboardX import SummaryWriter
from utils.utilization import mkdir
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import cv2


def adjust_learning_rate(init_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model_type = opt.model
    dataset = opt.dataset
    dimension = opt.dim
    MultitaskAutoEncoder = MAE2D
    print('Training %dD network'%dimension)
    slice_num = opt.thickness

    if dataset == 'retouch':
        train_dataset = h5RETOUCH('../datasets/retouch/train.h5', slice_num=slice_num, crop_size=opt.crop_size,
                                  match=False)
        test_dataset = h5RETOUCH('../datasets/retouch/val.h5', slice_num=slice_num, crop_size=opt.crop_size)

        save_format = 'mhd'
        res_block = 6
        evaluate_domain = evaluate_domain_3d
        activate=nn.Tanh()

    elif dataset == 'ixi':
        train_dataset = h5IXI('../datasets/ixi/train_ne_robet2.h5', slice_num=slice_num, crop_size=opt.crop_size, match=False)
        test_dataset = h5IXI('../datasets/ixi/val_ne_robet2.h5', slice_num=slice_num, crop_size=256)

        save_format = 'nifti'
        res_block = 6
        evaluate_domain = evaluate_domain_3d
        activate=nn.Tanh()

    else:
        raise NotImplementedError

    train_len = train_dataset.__len__()
    test_len = test_dataset.__len__()
    ratio = train_len / (train_len + test_len)
    print('Training set: %d, Testing set: %d, ratio: %.5f'%(train_len, test_len, ratio))

    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=opt.batch_size, shuffle=True,
                                                    num_workers=12,
                                                    pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=1, shuffle=False,
                                                    num_workers=12,
                                                    pin_memory=True)


    net = MultitaskAutoEncoder(1,1, n_blocks=res_block, activate=activate)
    #print(net)
    net = init_net(net, opt.init_type, 0.02, opt.gpu_ids)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    if opt.continue_train:
        print('Continue training from %s'%(opt.checkpoints_dir))
        state_dict = torch.load(os.path.join(opt.checkpoints_dir, opt.name + '/latest.pth'), map_location=str(device))
        net.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    scheduler = get_scheduler(optimizer, opt)
    classification_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    recon_criterion = torch.nn.L1Loss(reduction='mean')
    #criterion = nn.MSELoss()

    summarypath = os.path.join(opt.checkpoints_dir, opt.name)
    writer = SummaryWriter(summarypath)
    mkdir(summarypath)
    net.cuda()
    net.train()
    best_acc = -1
    best_rec = 999.
    best_val_acc = -1
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        lr = optimizer.param_groups[0]['lr']
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        for iteration, data in enumerate(data_loader_train):
            xs = data['A'].cuda()
            ys = data['B'].cuda()
            b, c, w, h = xs.size()
            fake_label = torch.zeros((b, 1))
            real_label = torch.ones((b, 1))
            labels = torch.cat([fake_label, real_label], 0).cuda()
            inputs = torch.cat([xs, ys], 0)
            shuffle_idx = torch.randperm(2*b)
            shuffle_idx = shuffle_idx[:b]

            inputs = inputs[shuffle_idx]
            labels = labels[shuffle_idx]
            pred, feat, rec = net(inputs)

            class_loss = classification_criterion(pred, labels)
            recon_loss = recon_criterion(rec, inputs)
            loss = class_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_logits = torch.sigmoid(pred)
            pred_logits = (pred_logits > 0.5).float()

            pred_acc = (pred_logits == labels).sum().float() / (labels.size(0))
            print('Epoch %s iteration %s: Class loss: %f, Rec loss: %f, acc %f, best acc %f, lr: %f' % (epoch, iteration, class_loss.item(),
                                                   recon_loss.item(),  pred_acc.item(), best_val_acc, lr))

            step = iteration + epoch * len(data_loader_train)
            if step % 200 == 0:
                fig = plt.figure()
                randint = np.random.randint(inputs.size(0))
                in_ = inputs.detach().cpu().numpy()[randint][0]
                out = rec.detach().cpu().numpy()[randint][0]
                ax = plt.subplot(1,2,1)
                plt.imshow(cv2.resize(in_, (256, 256)), cmap='gist_gray'), plt.title('in'), plt.axis('off')
                ax = plt.subplot(1,2,2)
                plt.imshow(cv2.resize(out, (256, 256)), cmap='gist_gray'), plt.title('out'), plt.axis('off')
                writer.add_figure('train/recon', fig, step)
                writer.add_scalar('train/reconstruction', recon_loss, step)
                plt.close()



        scheduler.step(epoch)
        if epoch % opt.save_epoch_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, step))
            save_suffix = 'epoch%d' % epoch
            save_path = os.path.join(summarypath, save_suffix)
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda(opt.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)
