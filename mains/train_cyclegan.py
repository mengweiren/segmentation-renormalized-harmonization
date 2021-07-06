import os, sys

sys.path.append('../')
from data.example_loader import h5lodaer

from configs.train_options import TrainOptions

import torch
import torch.utils.data
from torch.autograd import Variable

from models.cycle_gan import CycleGAN

from tensorboardX import SummaryWriter
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
from utils.utilization import mkdir, mkdirs
from utils.visualization import tensorboard_vis, create_group_fig
from utils.evaluation import eval_segmentation_batch
from models.networks2d import SingleSemanticDropout, PairedSemanticDropout
import shutil

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model_type = opt.model
    slice_num = opt.thickness
    dataset = opt.dataset
    dimension = opt.dim
    summarypath = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(summarypath):
        mkdirs(summarypath)

    assert slice_num == opt.input_nc

    train_dataset = h5lodaer('../datasets/train.h5', crop_size=opt.crop_size, src_key='scanner1', trg_key='scanner2')
    shutil.copy('../scripts/train_cyclegan.sh',
                os.path.join(summarypath, 'train_cyclegan.sh'))  # copy config file to output folder
    print('Training set: %d' % (train_dataset.__len__()))
    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=opt.batch_size, shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True)
    model = CycleGAN(opt)
    model.save_dir = summarypath

    writer = SummaryWriter(summarypath)
    if opt.continue_train:
        opt.epoch_count = opt.load_epoch

    model.setup(opt)
    model.train()
    if opt.sem_dropout:
        SM = PairedSemanticDropout(dropout_rate=0.3, num_class=4)
    for epoch in range(opt.epoch_count, opt.epoch_count + opt.niter + opt.niter_decay + 1):
        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count> + <save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for iteration, data in enumerate(data_loader_train):
            if opt.sem_dropout:
                data['A'], data['A_seg'], data['B'], data['B_seg'] = SM(data['A'], data['A_seg'], data['B'],
                                                                        data['B_seg'])
            model.set_input(data)
            print(data['A'].min(), data['A'].max(), data['B'].min(), data['B'].max())
            step = iteration + (epoch - 1) * len(data_loader_train)
            model.optimize_parameters(step)
            losses = model.get_current_losses()

            print('Src %s, Trg %s, Step %s [Epoch %s/%s Iteration %s/%s] (lr G/D %.5f %.5f):' % (src_scan['prefix'], trg_scan['prefix'],\
            step, epoch, opt.niter + opt.niter_decay,
            iteration, len(data_loader_train),
            model.optimizer_G.param_groups[0]['lr'],
            model.optimizer_D.param_groups[0]['lr']), losses)

            if step % 100 == 0:
                print('TENSORBOARD VISUALIZATION')
                writer.add_scalar('train/Loss_G_A', losses['G_A'], step)
                writer.add_scalar('train/Loss_D_A', losses['D_A'], step)
                writer.add_scalar('train/Loss_G_B', losses['G_B'], step)
                writer.add_scalar('train/Loss_D_B', losses['D_B'], step)
                writer.add_scalar('train/Loss_cycle_A', losses['cycle_A'], step)
                writer.add_scalar('train/Loss_cycle_B', losses['cycle_B'], step)
                if opt.lambda_identity > 0:
                    writer.add_scalar('train/Loss_idt_A', losses['idt_A'], step)
                    writer.add_scalar('train/Loss_idt_B', losses['idt_B'], step)
                if opt.lambda_cc > 0:
                    writer.add_scalar('train/cc_AB', losses['cc_AB'], step)
                    writer.add_scalar('train/cc_BA', losses['cc_BA'], step)

                if opt.spade and opt.joint_seg:
                    writer.add_scalar('train/seg_A_realA', losses['seg_real_A'], step)
                    writer.add_scalar('train/seg_A_fakeA', losses['seg_fake_A'], step)
                    # writer.add_scalar('train/seg_fake_A', losses['seg_fake_A'], step)
                    writer.add_scalar('train/seg_B_realB', losses['seg_real_B'], step)

                randind = np.random.randint(model.real_A.size(0))
                real_A = model.real_A.detach().cpu().numpy()[randind][0]
                real_B = model.real_B.detach().cpu().numpy()[randind][0]
                fake_A = model.fake_A.detach().cpu().numpy()[randind][0]
                fake_B = model.fake_B.detach().cpu().numpy()[randind][0]
                rec_A = model.rec_A.detach().cpu().numpy()[randind][0]
                rec_B = model.rec_B.detach().cpu().numpy()[randind][0]
                print('Visualizing CycleGAN', fake_B.min(), fake_B.max())

                imgs_vis = [real_A, fake_B, rec_A, real_B, fake_A, rec_B]
                imgs_titles = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']

                writer = tensorboard_vis(summarywriter=writer, step=step, board_name='train_new/cycle_crops',
                                         num_row=2, img_list=imgs_vis, cmaps='gist_gray', titles=imgs_titles,
                                         resize=True)


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


                if opt.joint_seg:
                    # fig = plt.figure()
                    cmap = matplotlib.colors.ListedColormap(['black', 'red', 'green', 'blue'])
                    print('Visualizing Segmentation')
                    seg_A_gt = model.seg_A.detach().cpu().numpy()[randind].transpose(1, 2, 0)
                    seg_B_gt = model.seg_B.detach().cpu().numpy()[randind].transpose(1, 2, 0)  # w, h, 4
                    if opt.prob_seg:
                        seg_A_gt, seg_B_gt = seg_A_gt[:, :, 1:], seg_B_gt[:, :, 1:]
                    else:
                        seg_A_gt = np.argmax(seg_A_gt, axis=-1).astype(np.float32)
                        seg_B_gt = np.argmax(seg_B_gt, axis=-1).astype(np.float32)

                    real_A_seg_out = post_process_seg_out(model.seg_real_A_out, randind, opt.prob_seg)
                    real_B_seg_out = post_process_seg_out(model.seg_real_B_out, randind, opt.prob_seg)
                    fake_A_seg_out = post_process_seg_out(model.seg_fake_A_out, randind, opt.prob_seg)
                    fake_B_seg_out = post_process_seg_out(model.seg_fake_B_out, randind, opt.prob_seg)

                    img_seg_vis = [real_A, real_A_seg_out, fake_B, fake_B_seg_out, seg_A_gt,
                                   real_B, real_B_seg_out, fake_A, fake_A_seg_out, seg_B_gt]
                    img_seg_tis = ['real_A', 'real_A_seg', 'fake_B', 'fake_B_seg', 'seg_A_gt',
                                   'real_B', 'real_B_seg', 'fake_A', 'fake_A_seg', 'seg_B_gt']
                    img_seg_map = [cmap if 'seg' in s else 'gist_gray' for s in img_seg_tis]

                    writer = tensorboard_vis(writer, step, 'train_new/seg_cycle_crops',
                                             num_row=2, img_list=img_seg_vis, cmaps=img_seg_map, titles=img_seg_tis,
                                             resize=True)

                    '''
                    Batch Evaluation
                    '''
                    print('----------src_seg_on_src_img----------- ')
                    acc_real_A, pre_real_A, rec_real_A, f1_real_A = eval_segmentation_batch(model.seg_real_A_out,
                                                                                            model.seg_A, opt.dataset)
                    print('----------trg_seg_on_hrm_img----------- ')
                    acc_fake_B, pre_fake_B, rec_fake_B, f1_fake_B = eval_segmentation_batch(model.seg_fake_B_out,
                                                                                            model.seg_A, opt.dataset)
                    print('----------trg_seg_on_trg_img----------- ')
                    acc_real_B, pre_real_B, rec_real_B, f1_real_B = eval_segmentation_batch(model.seg_real_B_out,
                                                                                            model.seg_B, opt.dataset)
                    # print('----------trg_seg_on_src_img----------- ')
                    # acc_fake_A, pre_fake_A, rec_fake_A, f1_fake_A = eval_segmentation_batch(model.seg_real_A_out, model.seg_A, opt.dataset)

        model.update_learning_rate()
        if (epoch) % opt.save_epoch_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving model (epoch %d, total_iters %d)' % (epoch, step))
            save_suffix = 'epoch{}step{}'.format(epoch, step)
            model.save_networks(save_suffix)

        if (epoch + 1) % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, step))
            save_suffix = 'latest'
            model.save_networks(save_suffix)


