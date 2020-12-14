import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import networks3d, networks2d
from .base_model import BaseModel
import itertools


class CycleGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        if opt.lambda_cc > 0:
            self.loss_names.append('cc_AB')
            self.loss_names.append('cc_BA')
        if opt.lambda_tv > 0:
            self.loss_names.append('tv_AB')
            self.loss_names.append('tv_BA')
        self.spade = opt.spade
        self.learn_seg = opt.joint_seg
        if opt.joint_seg:
            self.loss_names.append('seg_real_A')
            self.loss_names.append('seg_fake_A')
            self.loss_names.append('seg_real_B')
            self.loss_names.append('seg_fake_B')
        self.sem_dropout = opt.sem_dropout
        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.loss_names.append('idt_A')
            self.loss_names.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.sigma = opt.noise_std
        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        if opt.dim == 2:
            networks = networks2d
        else:
            networks = networks3d

        # define networks
        film_nc = opt.seg_nc
        if opt.mask:
            film_nc = opt.seg_nc + 1
            self.add_mask = True
        else:
            self.add_mask = False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ndf, opt.typeG, norm=opt.norm,
                                        use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=opt.spade,seg_nc=film_nc)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ndf, opt.typeG, norm=opt.norm,
                                        use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=opt.spade,seg_nc=film_nc)
        if self.learn_seg:
            if opt.dataset == 'msseg':
                stype = 'unet_up'
            else:
                stype = 'resunet'
            #stype = 'resunet'
            self.model_names.append('S_A')
            self.model_names.append('S_B')
            self.netS_A = networks.define_G(opt.input_nc, opt.seg_nc, opt.ngf, netG=stype, norm=opt.norm,
                                        use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=False, is_seg=True)
            self.netS_B = networks.define_G(opt.input_nc, opt.seg_nc, opt.ngf, netG=stype, norm=opt.norm,
                                        use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                        init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, return_feature=opt.fid, spade=False, is_seg=True)

        if self.isTrain:  # define discriminators
            self.multiD = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            n_layer=opt.n_layers_D, norm=opt.norm, init_type=opt.init_type,
                                            init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            n_layer=opt.n_layers_D, norm=opt.norm, init_type=opt.init_type,
                                            init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss(reduction='mean')
            self.criterionIdt = torch.nn.L1Loss(reduction='mean')
            if self.learn_seg:
                if opt.dataset == 'ixi' or opt.dataset == 'adni': bg_lambda = 0.5
                if opt.dataset == 'retouch' or opt.dataset=='msseg': bg_lambda = 0.7
                neg_w, pos_w = 0.2, 0.8
                self.criterionSeg = nn.CrossEntropyLoss(weight=torch.tensor([neg_w, pos_w]).cuda())
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

            if self.learn_seg:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),
                                                                      self.netS_A.parameters(), self.netS_B.parameters()),
                                                       lr=opt.lr_g, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr_g, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)



    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        if self.learn_seg:
            self.seg_A = input['A_seg'].to(self.device)
            self.seg_B = input['B_seg'].to(self.device)
        else:
            self.seg_A = None
            self.seg_B = None
        if self.add_mask:
            self.mask_A = input['A_mask'].to(self.device)
            self.mask_B = input['B_mask'].to(self.device)
        if self.sem_dropout:
            SM = networks2d.PairedSemanticDropout(0.3, 4)
            self.real_A, self.seg_A, self.real_B, self.seg_B = SM(self.real_A, self.seg_A, self.real_B, self.seg_B)

    def forward(self):

        if self.learn_seg:
            if self.spade:
                # segmentation conditioned cycle GAN with segmentation learnable
                self.seg_real_A_out = self.netS_A(self.real_A) #
                self.seg_real_B_out = self.netS_B(self.real_B)
                if self.add_mask:
                    seg_cond_A = torch.cat([self.seg_real_A_out, self.mask_A], dim=1)
                    seg_cond_B = torch.cat([self.seg_real_B_out, self.mask_B], dim=1)
                else:
                    seg_cond_A = self.seg_real_A_out
                    seg_cond_B = self.seg_real_B_out
                self.fake_B, _, _, _ = self.netG_A(self.real_A, seg_cond_A, freeze=False)
                self.fake_A, _, _, _ = self.netG_B(self.real_B, seg_cond_B, freeze=False)

                self.seg_fake_B_out = self.netS_B(self.fake_B)#self.tmpS_B(self.fake_B)#.detach()
                self.seg_fake_A_out = self.netS_A(self.fake_A)#self.tmpS_A(self.fake_A)#.detach()

                if self.add_mask:
                    seg_cond_A_fake = torch.cat([self.seg_fake_B_out, self.mask_A], dim=1)
                    seg_cond_B_fake = torch.cat([self.seg_fake_A_out, self.mask_B], dim=1)
                else:
                    seg_cond_A_fake = self.seg_fake_B_out
                    seg_cond_B_fake = self.seg_fake_A_out

                self.rec_A, _, _, _ = self.netG_B(self.fake_B, seg_cond_A_fake, freeze=True)
                self.rec_B, _, _, _ = self.netG_A(self.fake_A, seg_cond_B_fake, freeze=True)

            else:
                # cycle GAN with segmentation loss
                self.fake_B = self.netG_A(self.real_A)
                self.rec_A = self.netG_B(self.fake_B)
                self.fake_A = self.netG_B(self.real_B)
                self.rec_B = self.netG_A(self.fake_A)
                self.seg_real_A_out = self.netS_A(self.real_A)
                self.seg_real_B_out = self.netS_B(self.real_B)
                self.seg_fake_B_out = self.netS_B(self.fake_B)
                self.seg_fake_A_out = self.netS_A(self.fake_A)

            if self.sem_dropout and self.isTrain: # ONLY MASK OUT TO COMPUTE LOSS DURING TRAINING!!
                self.seg_real_A_out = self.seg_real_A_out * self.seg_A
                self.seg_real_B_out = self.seg_real_B_out * self.seg_B
                self.seg_fake_A_out = self.seg_fake_A_out * self.seg_B
                self.seg_fake_B_out = self.seg_fake_B_out * self.seg_A

                img_mask_A = torch.sum(self.seg_A, dim=1, keepdim=True)
                img_mask_B = torch.sum(self.seg_B, dim=1, keepdim=True)

                self.fake_B = self.fake_B * img_mask_A
                self.fake_A = self.fake_A * img_mask_B
                self.rec_A = self.rec_A * img_mask_A
                self.rec_B = self.rec_B * img_mask_B


        else:
            self.fake_B = self.netG_A(self.real_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B)
            self.rec_B = self.netG_A(self.fake_A)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        #print(pred_real.size())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self, steps):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        if self.multiD:
            self.loss_G_A = 0.5 * self.loss_G_A + 0.2 * self.criterionGAN(self.netD_A_pixel(self.fake_B), True) \
                            + 0.3 * self.criterionGAN(self.netD_A_34(self.fake_B), True)
            self.loss_G_B = 0.5 * self.loss_G_B + 0.2 * self.criterionGAN(self.netD_B_pixel(self.fake_A), True) \
                            + 0.3 * self.criterionGAN(self.netD_A_34(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        loss_G_B = self.loss_G_B + self.loss_cycle_B + self.loss_idt_B
        loss_G_A = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A

        if self.learn_seg:
            self.loss_seg_real_A = self.criterionSeg(self.seg_real_A_out, torch.argmax(self.seg_A.detach(), dim=1).long())
            self.loss_seg_real_B = self.criterionSeg(self.seg_real_B_out, torch.argmax(self.seg_B.detach(), dim=1).long())
            # print('Pretrained loss B', self.loss_seg_real_B.item())
            self.loss_seg_fake_A = self.criterionSeg(self.seg_fake_A_out, torch.argmax(self.seg_B.detach(), dim=1).long())
            self.loss_seg_fake_B = self.criterionSeg(self.seg_fake_B_out, torch.argmax(self.seg_A.detach(), dim=1).long())
            if self.spade:
                flag = 1.#(steps > 100)
            else:
                flag = 1.
            loss_G_A += flag * (self.loss_seg_real_A*0.5 + self.loss_seg_fake_B*0.05)
            loss_G_B += flag * (self.loss_seg_real_B*0.5 + self.loss_seg_fake_A*0.05)  # (self.loss_seg_real_B + self.loss_seg_fake_A)
            # loss_G_B += (self.loss_seg_real_B+ self.loss_seg_fake_A)
        loss_G = loss_G_A + loss_G_B
        return loss_G

    def optimize_parameters(self, steps):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.

        # G_A, S_A and G_B, S_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G(steps)
        loss_G.backward()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def add_channelwise_noise(self, in_tensor, sigma):
        noisy_image = torch.zeros(list(in_tensor.size())).data.normal_(0, sigma).cuda() + in_tensor
        # noisy_tensor = 2 * (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()) - 1
        return noisy_image
