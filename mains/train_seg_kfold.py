import os, sys
sys.path.append('../')
from data.retouch_loader import h5RETOUCH, TestRETOUCH
from data.ixi_loader import h5IXI, TestIXI
from data.msseg_loader import msseg, msseg_test, msseg_k_fold, msseg_k_fold_test
from data.msseg_h5 import scanner1, scanner2, scanner3
from configs.train_options import TrainOptions
from mains.test2d import evaluate_domain_seg
import itertools

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.networks2d import define_G

from tensorboardX import SummaryWriter
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from utils.visualization import tensorboard_vis
from utils.evaluation import eval_segmentation_batch
import shutil
from utils.utilization import append_dict, mkdirs
import nibabel as nib


def seg_loss(criterion, output, seg_map, lambda_fg=0.95, weight=1.):
    '''
    :param output: the segmentation before softmax!
    :param seg_map: one hot segmentation map with discrete range [0,1]
    :param weight: weight applied to the returned loss value
    :return:
    '''
    b, c, w, h = seg_map.size()
    seg_gt = torch.argmax(seg_map, dim=1)  # b, w, h
    mask = (seg_gt > 0.)
    mask_output = mask.float().view(b, 1, w, h).expand_as(output) * output
    mask_gt = mask.long() * seg_gt
    foreground_loss = criterion(mask_output, mask_gt)

    mask_back = (seg_gt == 0.)
    mask_back_output = mask_back.float().view(b, 1, w, h).expand_as(output) * output
    mask_back_gt = mask_back.long() * seg_gt
    background_loss = criterion(mask_back_output, mask_back_gt)
    # print(mask_output.size(), seg_gt.size())
    return weight * (lambda_fg * foreground_loss + (1-lambda_fg) * background_loss)
    #return foreground_loss, background_loss

def cross_entropy(pred, label):
    pred  = F.softmax(pred, dim=1)
    pred  = torch.clamp(pred,  1e-7, 1.0)
    label = torch.clamp(label, 1e-4, 1.0)
    return -1 * torch.sum(label * torch.log(pred), dim=1).mean()


def focal_loss(pred, label, alpha=1, gamma=2):
    ce_loss = F.cross_entropy(pred, label, reduction='none')  # important to add reduction='none' to keep per-batch-item loss
    #print(ce_loss.size())
    pt = torch.exp(-ce_loss)
    loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    return loss

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model_type = opt.model
    slice_num = opt.thickness
    dataset = opt.dataset
    dimension = opt.dim
    if dimension == 2:
        assert slice_num == opt.input_nc
    summarypath = os.path.join(opt.checkpoints_dir, opt.name+'/fold_%d'%opt.fold)
    if not os.path.exists(summarypath):
        mkdirs(summarypath)
    shutil.copy('../scripts/train_seg.sh', os.path.join(summarypath, 'train_seg.sh'))  # copy config file to output folder

    best_src = -1
    if dataset == 'msseg':
        labels_dict = ['BG', 'Lesion']
        #scan = scanner1
        src_scan, trg_scan = scanner3, scanner3
        mni = True
        fold_k = int(opt.fold)
        print(fold_k)
        folder = '../datasets/mni_msseg_all/'
        folder = '../datasets/msseg/'
        pad_x, pad_y = 256, 256
        train_dataset = msseg_k_fold(folder, src=src_scan, trg=trg_scan, is_train=True, crop_size=opt.crop_size,
                                     resize=None, mni=True, fold=fold_k, normalize='self')
        val_dataset = msseg_k_fold(folder, src=src_scan, trg=trg_scan, is_train=False, crop_size=opt.crop_size,
                                     resize=None, mni=True, fold=fold_k, normalize='self')
        with open(os.path.join(summarypath, 'train_%s_fold_%d.txt'%(src_scan['prefix'], fold_k)), 'w') as f:
            f.writelines("%s\n" % s for s in train_dataset.src_sub_list)
        with open(os.path.join(summarypath, 'val_%s_fold_%d.txt'%(src_scan['prefix'], fold_k)), 'w') as f:
            f.writelines("%s\n" % test_dataset.sub_)
    else:
        raise NotImplementedError

    data_loader_train = torch.utils.data.DataLoader(dataset= train_dataset,
                                               batch_size=opt.batch_size, shuffle=True,
                                               num_workers=12,
                                               pin_memory=True)
    data_loader_val = torch.utils.data.DataLoader(dataset= val_dataset,
                                               batch_size=4, shuffle=True,
                                               num_workers=12,
                                               pin_memory=True)

    segnet = define_G(opt.input_nc, opt.seg_nc, opt.ngf, netG=opt.typeG, norm=opt.norm,
                    use_dropout=not opt.no_dropout, init_type=opt.init_type,
                    init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=False, is_seg=True)

    neg_w, pos_w = 0.2, 0.8
    seg_criterion = nn.CrossEntropyLoss(weight=torch.tensor([neg_w, pos_w]).cuda())

    optimizerA = torch.optim.Adam(segnet.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    writer = SummaryWriter(summarypath)
    if opt.continue_train:
        opt.epoch_count = opt.load_epoch
        segnet.load_state_dict(torch.load(os.path.join(summarypath, 'latestnet_S_A.pth')))

    segnet.cuda()
    segnet.train()

    for epoch in range(opt.epoch_count, opt.epoch_count+opt.niter + opt.niter_decay + 1):
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count> + <save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for iteration, data in enumerate(data_loader_train):
            xs = data['A'].cuda()
            seg_x = data['A_seg'].cuda()
            seg_x_long = torch.argmax(seg_x, dim=1).long()

            #print(data['A'].min(), data['A'].max(), data['B'].min(), data['B'].max())
            seg_x_pred = segnet(xs)
            seg_x_loss = seg_criterion(seg_x_pred, seg_x_long) 
            step = iteration + (epoch-1) * len(data_loader_train)
            optimizerA.zero_grad()
            seg_x_loss.backward()
            optimizerA.step()

            print('Fold %d on Scan %s Step %s [Epoch %s|%s Iteration %s]: Seg Loss: %.5f, Best: %.5f'
                  %(fold_k, src_scan['prefix'], step, epoch, opt.epoch_count+opt.niter + opt.niter_decay + 1, iteration, seg_x_loss.item(), best_src))


            def post_process_seg_out(net_output, batch_idx, prob_seg):
                if prob_seg is False:
                    seg_out = torch.softmax(net_output, dim=1)
                    seg_out = seg_out.detach().cpu().numpy()[batch_idx].transpose(1, 2, 0)
                    seg_vis = np.argmax(seg_out, axis=-1).astype(np.float32)
                else:
                    seg_out = torch.softmax(net_output, dim=1)
                    seg_vis = seg_out.detach().cpu().numpy()[batch_idx].transpose(1, 2, 0)
                    seg_vis = seg_vis[:, :, 1:]
                return seg_vis

            if step % 50 == 0:
                randind = np.random.randint(xs.size(0))

                fig = plt.figure()
                real_A = xs.detach().cpu().numpy()[randind][0]
                print('Visualizing Segmentation')
                seg_A_gt = seg_x.detach().cpu().numpy()[randind].transpose(1, 2, 0)  # w, h, 4
                if opt.prob_seg:
                    seg_B_gt =  seg_B_gt[:, :, 1:]
                else:
                    seg_A_gt = np.argmax(seg_A_gt, axis=-1).astype(np.float32)

                real_A_seg_out = post_process_seg_out(seg_x_pred, randind, opt.prob_seg)
                cmap = matplotlib.colors.ListedColormap(['black', 'red', 'green', 'blue'])
                img_seg_vis = [real_A, real_A_seg_out, seg_A_gt]
                img_seg_tis = ['real_A', 'real_A_seg', 'seg_A_gt']
                img_seg_map = [cmap if 'seg' in s else 'gist_gray' for s in img_seg_tis]

                writer = tensorboard_vis(writer, step, 'train_seg_%s/crops'%src_scan['prefix'],
                                         num_row=1, img_list=img_seg_vis, cmaps=img_seg_map, titles=img_seg_tis,
                                         resize=True)

        if epoch % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, step))
                save_path = os.path.join(summarypath, 'latest')
                torch.save(segnet.cpu().state_dict(), save_path + 'net_S_A.pth')
                segnet.cuda(opt.gpu_ids[0])

        if epoch % opt.save_epoch_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving model (epoch %d, total_iters %d)' % (epoch, step))
            save_suffix = 'epoch%d' % epoch
            save_path = os.path.join(summarypath, save_suffix)
            torch.save(segnet.cpu().state_dict(), save_path + 'net_S_A.pth')
            segnet.cuda(opt.gpu_ids[0])

    exit()