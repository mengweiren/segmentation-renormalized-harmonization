import os, sys
sys.path.append('../')
from data.retouch_loader import h5RETOUCH, TestRETOUCH
from data.ixi_loader import h5IXI, TestIXI
from configs.train_options import TrainOptions

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from models.networks2d import define_G

from tensorboardX import SummaryWriter
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.visualization import tensorboard_vis
from utils.evaluation import eval_segmentation_batch


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


if __name__ == '__main__':
    opt = TrainOptions().parse()
    model_type = opt.model
    slice_num = opt.thickness
    dataset = opt.dataset
    dimension = opt.dim
    if dimension == 2:
        assert slice_num == opt.input_nc

    best_src = -1
    best_trg = -1
    if dataset == 'retouch':
        train_dataset = h5RETOUCH('../datasets/retouch/train.h5', slice_num=slice_num, crop_size=opt.crop_size)
        test_dataset = h5RETOUCH('../datasets/retouch/val.h5', slice_num=slice_num, crop_size=opt.crop_size)

    elif dataset == 'ixi':
        labels_dict = ['BG', 'CSF', 'Grey matter', 'White matter']
        train_dataset = h5IXI('../datasets/ixi/train_ne_prob.h5', slice_num=slice_num, crop_size=opt.crop_size, prob_seg=True)
        test_dataset = h5IXI('../datasets/ixi/val_ne_prob.h5', slice_num=slice_num, crop_size=opt.crop_size, prob_seg=True)

    else:
        raise NotImplementedError

    data_loader_train = torch.utils.data.DataLoader(dataset= train_dataset,
                                               batch_size=opt.batch_size, shuffle=True,
                                               num_workers=12,
                                               pin_memory=True)

    segnetA = define_G(opt.input_nc, opt.seg_nc, opt.ngf, netG=opt.typeG, norm=opt.norm,
                                        use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=False, is_seg=True)
    segnetB = define_G(opt.input_nc, opt.seg_nc, opt.ngf, netG=opt.typeG, norm=opt.norm,
                                        use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=False, is_seg=True)

    seg_criterion = nn.CrossEntropyLoss()

    optimizerA = torch.optim.Adam(segnetA.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    optimizerB = torch.optim.Adam(segnetB.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))


    summarypath = os.path.join(opt.checkpoints_dir, opt.name)
    writer = SummaryWriter(summarypath)
    if opt.continue_train:
        opt.epoch_count = opt.load_epoch
        segnetA.load_state_dict(torch.load(os.path.join(summarypath, 'latestnet_S_A.pth')))
        segnetB.load_state_dict(torch.load(os.path.join(summarypath, 'latestnet_S_B.pth')))

    segnetA.cuda()
    segnetA.train()

    segnetB.cuda()
    segnetB.train()
    for epoch in range(opt.epoch_count, opt.epoch_count+opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for iteration, data in enumerate(data_loader_train):
            xs, ys = data['A'].cuda(), data['B'].cuda()
            seg_x, seg_y = data['A_seg'].cuda(), data['B_seg'].cuda()
            seg_x_pred = segnetA(xs)
            seg_y_pred = segnetB(ys)

            seg_x_loss = cross_entropy(pred=seg_x_pred, label=seg_x)
            seg_y_loss = cross_entropy(pred=seg_y_pred, label=seg_y)

            step = iteration + (epoch-1) * len(data_loader_train)
            optimizerA.zero_grad()
            seg_x_loss.backward()
            optimizerA.step()

            optimizerB.zero_grad()
            seg_y_loss.backward()
            optimizerB.step()

            print('Step %s [Epoch %s Iteration %s]: Seg X Loss: %.5f, Seg Y Loss: %.5f'%\
                  (step, epoch, iteration, seg_x_loss.item(), seg_y_loss.item()))


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
                randind = np.random.randint(ys.size(0))

                fig = plt.figure()
                real_A = xs.detach().cpu().numpy()[randind][0]
                real_B = ys.detach().cpu().numpy()[randind][0]

                print('Visualizing Segmentation')
                seg_A_gt = seg_x.detach().cpu().numpy()[randind].transpose(1, 2, 0)  # w, h, 4
                seg_B_gt = seg_y.detach().cpu().numpy()[randind].transpose(1, 2, 0)  # w, h, 4
                if opt.prob_seg:
                    seg_B_gt =  seg_B_gt[:, :, 1:]
                else:
                    seg_A_gt = np.argmax(seg_A_gt, axis=-1).astype(np.float32)
                    seg_B_gt = np.argmax(seg_B_gt, axis=-1).astype(np.float32)

                real_A_seg_out = post_process_seg_out(seg_x_pred, randind, opt.prob_seg)
                real_B_seg_out = post_process_seg_out(seg_y_pred, randind, opt.prob_seg)
                cmap = matplotlib.colors.ListedColormap(['black', 'red', 'green', 'blue'])
                img_seg_vis = [real_A, real_A_seg_out, seg_A_gt, real_B, real_B_seg_out, seg_B_gt]
                img_seg_tis = ['real_A', 'real_A_seg', 'seg_A_gt', 'real_B', 'real_B_seg', 'seg_B_gt']
                img_seg_map = [cmap if 'seg' in s else 'gist_gray' for s in img_seg_tis]

                writer = tensorboard_vis(writer, step, 'train_seg/crops',
                                         num_row=2, img_list=img_seg_vis, cmaps=img_seg_map, titles=img_seg_tis,
                                         resize=True)

                '''
                Batch Evaluation
                '''
                print('----------src_on_src_img----------- ')
                acc_real_A, pre_real_A, rec_real_A, f1_real_A = eval_segmentation_batch(seg_x_pred,
                                                                                        seg_x, opt.dataset)

                print('----------trg_on_trg_img----------- ')
                acc_real_B, pre_real_B, rec_real_B, f1_real_B = eval_segmentation_batch(seg_y_pred,
                                                                                        seg_y, opt.dataset)

                writer.add_scalar('train_seg_rec/f1_x', f1_real_A[labels_dict[1]], step)
                writer.add_scalar('train_seg_rec/f1_y', f1_real_B[labels_dict[1]], step)

        if epoch % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, step))
            save_path = os.path.join(summarypath, 'latest')
            torch.save(segnetA.cpu().state_dict(), save_path + 'net_S_A.pth')
            segnetA.cuda(opt.gpu_ids[0])
            torch.save(segnetB.cpu().state_dict(), save_path + 'net_S_B.pth')
            segnetB.cuda(opt.gpu_ids[0])


        if epoch % opt.save_epoch_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving model (epoch %d, total_iters %d)' % (epoch, step))
            save_suffix = 'epoch%d' % epoch
            save_path = os.path.join(summarypath, save_suffix)
            torch.save(segnetA.cpu().state_dict(), save_path + 'net_S_A.pth')
            segnetA.cuda(opt.gpu_ids[0])
            torch.save(segnetB.cpu().state_dict(), save_path + 'net_S_B.pth')
            segnetB.cuda(opt.gpu_ids[0])