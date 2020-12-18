import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.normalization import SPADE, SegFeature, SharedSPADE
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import random
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight, 1.0, init_gain)
            init.constant_(m.bias, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        #assert(torch.cuda.is_available())
        net.cuda()#to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf=64, netG='resunet', norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[], return_feature=False, spade=False, is_seg=False, seg_nc=4):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'unet_up':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, upsample=True)
    elif netG == 'unet':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, upsample=False)
    elif netG == 'resunet':
        if spade:
            net = SharedResSpadeUnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, label_nc=seg_nc)
        else:
            net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer,
                                use_dropout=use_dropout, upsample=False, res=True, is_seg=is_seg)
    elif netG == 'resunet_seg':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer,
                            use_dropout=use_dropout, upsample=False, res=True, is_seg=True)
    elif netG == 'resunet_up':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, upsample=True, res=True)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD='n_layers', n_layer=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layer, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

def symmetric_cross_entropy(pred, label, alpha, beta):
    pred  = F.softmax(pred, dim=1)
    pred  = torch.clamp(pred, 1e-7, 1.0)
    label = torch.clamp(label, 1e-4, 1.0)
    ce_loss  = -1 * torch.sum(label * torch.log(pred), dim=1).mean()
    rce_loss = -1 * torch.sum(pred  * torch.log(label), dim=1).mean()

    return alpha * ce_loss + beta * rce_loss


def cross_entropy(pred, label):
    pred  = F.softmax(pred, dim=1)
    pred  = torch.clamp(pred,  1e-7, 1.0)
    label = torch.clamp(label, 1e-4, 1.0)
    return -1 * torch.sum(label * torch.log(pred), dim=1).mean()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SingleSemanticDropout(nn.Module):
    def __init__(self, dropout_rate=0.2, num_class=4, weighted_prob=False):
        super(SingleSemanticDropout, self).__init__()
        self.p = dropout_rate
        self.nclass = num_class
        self.weighted_prob = weighted_prob

    def __call__(self, img, seg):
        #tmp_img = img
        #tmp_img = (tmp_img + 1)/2.
        label = torch.argmax(seg, dim=1)
        lab_list = list(label.unique().numpy())
        if np.random.uniform(0,1) <= self.p:
            if self.weighted_prob:
                weights = np.zeros(len(lab_list))
                for l in range(len(lab_list)):
                    weights[l] = (label == lab_list[l]).sum().float()/ label.nelement()
                weights = 1. - softmax(weights)
                #print(lab_list, weights)
                selected = random.choices(lab_list, weights=weights, k=len(lab_list))
            else:
                selected = random.choices(lab_list, k=len(lab_list))

            selected = np.unique(selected)
            #exclude = np.array(list(set(np.arange(self.nclass)) - set(selected)))
            #print(selected)
            mask = torch.zeros_like(seg)
            mask[:, selected, :, :] = seg[:, selected, :, :]

            mask_img = torch.sum(seg[:, selected, :, :], dim=1, keepdim=True) * img
            #mask_img = 2*mask_img - 1.
            #print(tmpa.min(), tmpa.max(), tmpb.min(), tmpb.max())
            assert mask_img.size() == img.size() , 'Masked image shape must match input'

            return mask_img, mask

        else:
            return img, seg


class PairedSemanticDropout(nn.Module):
    def __init__(self, dropout_rate=0.2, num_class=4, weighted_prob=False):
        super(PairedSemanticDropout, self).__init__()
        self.p = dropout_rate
        self.nclass = num_class
        self.weighted_prob = weighted_prob

    def __call__(self, img_a, seg_a, img_b, seg_b):
        label_a = torch.argmax(seg_a, dim=1)
        label_b = torch.argmax(seg_b, dim=1)
        if label_a.unique().size() > label_b.unique().size():
            tmp1, tmp2 = label_a.unique(), label_b.unique()
        else:
            tmp1, tmp2 = label_b.unique(), label_a.unique()
        label_ab = []
        for i in tmp1:
            if i in tmp2:
                label_ab.append(i.item())
        if np.random.uniform(0,1) <= self.p:
            if self.weighted_prob:
                weights = np.zeros(len(label_ab))
                for l in range(len(label_ab)):
                    weights[l] = ((label_a == label_ab[l]).sum() + (label_b == label_ab[l]).sum()).float()/ (label_a.nelement() + label_b.nelement())
                weights = 1. - softmax(weights)
                print(label_ab, weights)
                selected = random.choices(label_ab, weights=weights, k=len(label_ab))
            else:
                selected = random.choices(label_ab, k=len(label_ab))

            #exclude = np.array(list(set(np.arange(self.nclass)) - set(selected)))
            #print(selected)
            mask_a, mask_b = torch.zeros_like(seg_a), torch.zeros_like(seg_b)
            mask_a[:, selected, :, :] = seg_a[:, selected, :, :]
            mask_b[:, selected, :, :] = seg_b[:, selected, :, :]

            mask_img_a = torch.sum(seg_a[:, selected, :, :], dim=1, keepdim=True) * img_a
            mask_img_b = torch.sum(seg_b[:, selected, :, :], dim=1, keepdim=True) * img_b
            tmpa, tmpb = torch.sum(seg_a[:, selected, :, :], dim=1), torch.sum(seg_b[:, selected, :, :], dim=1)
            #print(tmpa.min(), tmpa.max(), tmpb.min(), tmpb.max())
            assert mask_img_a.size() == img_a.size() and mask_img_b.size() == img_b.size(), 'Masked image shape must match input'
            assert mask_a.size() == seg_a.size() and mask_b.size() == seg_b.size(), 'Masked image shape must match input'

            return mask_img_a, mask_a, mask_img_b, mask_b

        else:
            return img_a, seg_a, img_b, seg_b


class SegLoss(nn.Module):
    """
    :param output: segmentation before activation!! (No softmax here)
    :param target: one hot segmentation map with discrete range [0,1]
    :param weight: weight applied to the background
    :param type: one_hot -- segmentation map is scalar valued; prob -- segmentation map is a probability map
    :return:
    """
    def __init__(self, weight=0.5, prob_seg=False):
        super(SegLoss, self).__init__()
        self.prob_seg = prob_seg
        self.ce = nn.CrossEntropyLoss()
        self.weight = weight

    def __call__(self, prediction, target):
        b, c, w, h = target.size()
        if not self.prob_seg:
            seg_gt = torch.argmax(target, dim=1)  # b, w, h
            mask = (seg_gt > 0.)
            mask_output = mask.float().view(b, 1, w, h).expand_as(prediction) * prediction
            mask_gt = mask.long() * seg_gt
            foreground_loss = self.ce(mask_output, mask_gt)

            mask_back = (seg_gt == 0.)
            mask_back_output = mask_back.float().view(b, 1, w, h).expand_as(prediction) * prediction
            mask_back_gt = mask_back.long() * seg_gt
            background_loss = self.ce(mask_back_output, mask_back_gt)
            #print(foreground_loss, background_loss)
            return (1-self.weight)*foreground_loss + self.weight * background_loss

        else:
            return symmetric_cross_entropy(prediction, target, 1., 0.)#self.ce(prediction, target.float())



class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def correlation_coefficient_loss(real, fake):
    """Calculate the correlation coefficient"""
    var_real = real - torch.mean(real)
    var_fake = fake - torch.mean(fake)

    cc = torch.sum(var_real * var_fake) / (1e-5+(torch.sqrt(torch.sum(var_real ** 2)) * torch.sqrt(torch.sum(var_fake ** 2))))
    return cc


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    tv = tv_weight * (h_variance + w_variance)
    return tv

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class SharedResSpadeUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='zeros', label_nc=4):
        super(SharedResSpadeUnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.shared_seg_feature = SegFeature(label_nc=label_nc)
        self.head = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        #self.down_0 = nn.Conv2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_0 = SharedSPADEResBlock(1*ngf, 2*ngf)

        self.down_1 = nn.Conv2d(2*ngf, 2*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_1 = SharedSPADEResBlock(2*ngf, 4*ngf)

        self.down_2 = nn.Conv2d(4*ngf, 4*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_2 = SharedSPADEResBlock(4*ngf, 8*ngf)

        self.down_3 = nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_3 = SharedSPADEResBlock(8*ngf, 8*ngf)

        self.middle_0 = SharedSPADEResBlock(8*ngf, 8*ngf)
        self.middle_1 = SharedSPADEResBlock(8*ngf, 8*ngf)

        self.up = nn.Upsample(scale_factor=2)
        self.up_3 = SharedSPADEResBlock(16*ngf, 8*ngf)
        self.up_2 = SharedSPADEResBlock(16*ngf, 4*ngf)
        self.up_1 = SharedSPADEResBlock(8*ngf, 2*ngf)
        self.up_0 = SharedSPADEResBlock(4*ngf, 1*ngf)

        self.tail = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.activate = nn.Tanh()

    def set_required_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, input, seg):
        #self.set_required_grad(not freeze)
        seg_feat, seg_rep, seg_off, loss = self.shared_seg_feature(input, seg)
        x = self.head(input)  # n, w/2, h/2
        x = F.relu(x)
        x_0 = self.down_spade_0(x, seg_feat)  # 2n, w/2, h/2
        x_1 = self.down_spade_1(self.down_1(x_0), seg_feat)  # 4n, w/4, h/4
        x_2 = self.down_spade_2(self.down_2(x_1), seg_feat)  # 8n, w/8, h/8
        x_3 = self.down_spade_3(self.down_3(x_2), seg_feat)  # 8n, w/16, h/16
        out = self.up(x_3) # 8n, w/8, h/8
        out = self.middle_0(out, seg_feat)  #8n, w/8, h/8
        out = self.middle_1(out, seg_feat)  #8n, w/8, h/8
        #out = self.up(out)  # 8n, w/8, h/8
        out = self.up_2(torch.cat([out, x_2], 1), seg_feat)  #4n, w/8, w/8
        out = self.up(out) #4n, w/4, w/4
        out = self.up_1(torch.cat([out, x_1], 1), seg_feat)  #2n, w/4, w/4

        out = self.up(out)   #2n, w/2, h/2
        out = self.up_0(torch.cat([out, x_0], 1), seg_feat)  #n, w/2, w/2

        out = self.up(out)   #n, w, h
        out = self.tail(F.leaky_relu(out, 2e-1))

        out = self.activate(out)
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', activate=nn.Tanh):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.down_model = nn.Sequential(*model)
        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [activate]

        self.upmodel = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        feature = self.down_model(input)
        #print(feature.size())
        rec = self.upmodel(feature)
        return feature, rec


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResidualBasicBlock(nn.Module):
    def __init__(self, input_nc, output_nc, biases, padding, dropout, norm):
        super(ResidualBasicBlock, self).__init__()
        
        self.res_conv = ResnetBlock(input_nc, padding_type=padding, use_dropout=dropout, use_bias=biases, norm_layer=norm)
        self.down_conv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=biases)

    def forward(self, x):
        #print(x.size(), self.conv(x).size())
        out = self.res_conv(x)
        out = self.down_conv(out)
        return out


class SPADEGenerator(nn.Module):
    def __init__(self, opt, num_upsampling_layers='most', use_vae=False, z_dim=2048, norm_G='spadeinstance3x3'):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        semantic_nc = 4
        self.use_vae = use_vae
        self.sw, self.sh = self.compute_latent_vector_size(num_upsampling_layers, opt)
        self.num_upsampling_layers = num_upsampling_layers
        if self.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResBlock(16 * nf, 16 * nf, norm_G, semantic_nc)

        self.G_middle_0 = SPADEResBlock(16 * nf, 16 * nf, norm_G, semantic_nc)
        self.G_middle_1 = SPADEResBlock(16 * nf, 16 * nf, norm_G, semantic_nc)

        self.up_0 = SPADEResBlock(16 * nf, 8 * nf, norm_G, semantic_nc)
        self.up_1 = SPADEResBlock(8 * nf, 4 * nf, norm_G, semantic_nc)
        self.up_2 = SPADEResBlock(4 * nf, 2 * nf, norm_G, semantic_nc)
        self.up_3 = SPADEResBlock(2 * nf, 1 * nf, norm_G, semantic_nc)

        final_nc = nf

        if num_upsampling_layers == 'most':
            self.up_4 = SPADEResBlock(1 * nf, nf // 2, norm_G, semantic_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 1, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.activate = nn.Tanh()

    def compute_latent_vector_size(self, num_upsampling_layers, opt, aspect_ratio=1):
        if num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif num_upsampling_layers == 'more':
            num_up_layers = 6
        elif num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.num_upsampling_layers == 'more' or \
           self.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = self.activate(x)
        return x


class ResSpadeUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='zeros'):
        super(ResSpadeUnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.head = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        self.down_spade_0 = SPADEResBlock(1*ngf, 2*ngf)

        self.down_1 = nn.Conv2d(2*ngf, 2*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_1 = SPADEResBlock(2*ngf, 4*ngf)

        self.down_2 = nn.Conv2d(4*ngf, 4*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_2 = SPADEResBlock(4*ngf, 8*ngf)

        self.down_3 = nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.down_spade_3 = SPADEResBlock(8*ngf, 8*ngf)

        self.middle_0 = SPADEResBlock(8*ngf, 8*ngf)
        self.middle_1 = SPADEResBlock(8*ngf, 8*ngf)

        self.up = nn.Upsample(scale_factor=2)
        self.up_3 = SPADEResBlock(16*ngf, 8*ngf)
        self.up_2 = SPADEResBlock(16*ngf, 4*ngf)
        self.up_1 = SPADEResBlock(8*ngf, 2*ngf)
        self.up_0 = SPADEResBlock(4*ngf, 1*ngf)

        self.tail = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.activate = nn.Tanh()

    def forward(self, input, seg):
        x = self.head(input)  # n, w/2, h/2
        x = F.relu(x)
        x_0 = self.down_spade_0(x, seg)  # 2n, w/2, h/2
        x_1 = self.down_spade_1(self.down_1(x_0), seg)  # 4n, w/4, h/4
        x_2 = self.down_spade_2(self.down_2(x_1), seg)  # 8n, w/8, h/8
        x_3 = self.down_spade_3(self.down_3(x_2), seg)  # 8n, w/16, h/16
        out = self.up(x_3) # 8n, w/8, h/8
        out = self.middle_0(out, seg)  #8n, w/8, h/8
        out = self.middle_1(out, seg)  #8n, w/8, h/8
        #out = self.up(out)  # 8n, w/8, h/8
        out = self.up_2(torch.cat([out, x_2], 1), seg)  #4n, w/8, w/8
        out = self.up(out) #4n, w/4, w/4
        out = self.up_1(torch.cat([out, x_1], 1), seg)  #2n, w/4, w/4

        out = self.up(out)   #2n, w/2, h/2
        out = self.up_0(torch.cat([out, x_0], 1), seg)  #n, w/2, w/2

        out = self.up(out)   #n, w, h
        out = self.tail(F.leaky_relu(out, 2e-1))

        out = self.activate(out)
        return out, None, None, None



class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, upsample=False, res=False, is_seg=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, upsample=upsample, residual=res, is_seg=is_seg)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                                 upsample=upsample, residual=res, is_seg=is_seg)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, upsample=upsample, residual=res, is_seg=is_seg)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, upsample=upsample, residual=res, is_seg=is_seg)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             upsample=upsample, residual=res, is_seg=is_seg)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, upsample=upsample, residual=res, is_seg=is_seg)  # add the outermost layer

    def set_required_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, input, requires_grad=True):
        """Standard forward"""
        #print(self.model)
        #print(input.size())
        #print(seg.size())
        self.set_required_grad(requires_grad)
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 upsample=True, residual=False, padding_type='zeros', is_seg=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        kw = 4
        p = 0
        downconv = []
        upconv = []

        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
            upconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
            upconv += [nn.ReplicationPad2d(1)]
        else:
            p = 1
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        if residual is True:  # add residual connection
            downconv = [ResidualBasicBlock(input_nc, inner_nc, biases=use_bias, norm=norm_layer,
                                          dropout=use_dropout, padding='reflect')]
        else:
            downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                   stride=2, padding=p, bias=use_bias)]

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)



        if outermost:
            upconv += [nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kw, stride=2,
                                        padding=p)]
            if upsample:
                up_interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                conv = nn.Conv2d(inner_nc * 2, outer_nc,
                                kernel_size=3, padding=p)
                upconv = [up_interp, conv]
            down = downconv
            if is_seg is False:
                up = [uprelu] + upconv + [nn.Tanh()]
            else:
                print('No activation here')
                up = [uprelu] + upconv
            model = down + [submodule] + up


        elif innermost:
            upconv = [nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kw, stride=2,
                                        padding=p, bias=use_bias)]
            if upsample:
                up_interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                conv = nn.Conv2d(inner_nc, outer_nc, stride=1, kernel_size=3, padding=1)
                upconv = [up_interp, conv]
            down = [downrelu] + downconv
            up = [uprelu] + upconv + [upnorm]
            model = down + up
            #print(model)
        else:
            upconv += [nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kw, stride=2,
                                        padding=p, bias=use_bias)]
            if upsample:
                up_interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                conv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1)
                upconv = [up_interp, conv]

            down = [downrelu] + downconv + [downnorm]
            up = [uprelu]+ upconv +[upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(x.size())
        #return self.model(x, seg)
        if self.outermost:
            #print('Outermost', x.size(), self.model(x,seg).size())
            return self.model(x)
        else:   # add skip connections
            #print('Inner',x.size(), self.model(x).size())
            return torch.cat([x, self.model(x)], 1)



class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, norm_G='spadeinstance3x3', semantic_nc=4):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SharedSPADEResBlock(SPADEResBlock):
    def __init__(self, fin, fout, norm_G='spadeinstance3x3', semantic_nc=4):
        #super().__init__(fin, fout)
        SPADEResBlock.__init__(self, fin, fout, norm_G='spadeinstance3x3', semantic_nc=4)
        fmiddle = min(fin, fout)
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SharedSPADE(spade_config_str, norm_nc=fin, label_nc=semantic_nc)
        self.norm_1 = SharedSPADE(spade_config_str, norm_nc=fmiddle, label_nc=semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SharedSPADE(spade_config_str, fin, semantic_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                #norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            #norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        #print(self.model)  #receptive field 16
        #print(self.model(input).size())
        return self.model(input)



class ResMultitaskAutoEncoder(nn.Module):
    def __init__(self, input_nc, numclass, n_blocks=6, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, activate=nn.Tanh()):
        super(ResMultitaskAutoEncoder, self).__init__()
        self.net = ResnetGenerator(input_nc, input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, activate=activate)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(1024, numclass)
        )

    def forward(self, input):
            feature, rec = self.net(input)
            feat = self.avgpool(feature)
            pred = torch.flatten(feat, 1)
            #print(feat.size())
            pred = self.classifier(pred)
            return pred, feat, rec