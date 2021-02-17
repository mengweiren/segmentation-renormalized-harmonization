import argparse
import os
import torch
import models
import data
from utils.utilization import mkdir, mkdirs


class BaseOptions():
    """This class defines configs used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the configs.
    It also gathers additional configs defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common configs that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='cyclegan',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='../ckpts7', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan_3d',
                            help='chooses which model to use. [cycle_gan_3d | cycle_gan_2d_slice | test ]')
        parser.add_argument('--input_nc', type=int, default=1,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--f_map', type=list, default=[16, 32, 64, 128], help='# of gen filters in the last conv layer')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--typeG', type=str, default='unet',
                            help='specify generator architecture [unet | resunet ]')
        parser.add_argument('--n_layers_D', type=int, default=4, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--crop_size', type=int, default=16, help='then crop to this size')
        parser.add_argument('--thickness', type=int, default=3, help='thickness when doing the cropping')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--dim', type=int, default=2, help='2|3')
        parser.add_argument('--dataset', type=str, default='adni',help='select a dataset')
        parser.add_argument('--lambda_cc', type=float, default=0.5,help='use correlation coefficient loss if larger than 0')
        parser.add_argument('--lambda_tv', type=float, default=0.5,help='use total variance regularization if larger than 0')
        parser.add_argument('--fid', action='store_true',help='calculate frechet inception distance')
        parser.add_argument('--srenorm', action='store_true',help='using spatial adaptive denormalization')
        parser.add_argument('--joint_seg', action='store_true',help='learning segmentation instead of input segmentation map, and using spatial adaptive denormalization')
        parser.add_argument('--prob_seg', action='store_true',help='segmentation map is a probability')
        parser.add_argument('--load_epoch', type=int, default=0, help='continue training: the epoch to continue from')
        parser.add_argument('--load_step', type=int, default=0, help='continue training: the step to continue from')
        parser.add_argument('--sem_dropout', action='store_true', help='semantic dropout or not')
        parser.add_argument('--seg_nc', type=int, default=4, help='number of semantic class')
        parser.add_argument('--fold', type=float, default=0, help='fold id for LOOCV')
        parser.add_argument('--mask', action='store_true',help='add mask for brain')

        self.initialized = True
        return parser


    def gather_options(self):
        """Initialize our parser with basic configs(only once).
        Add additional model-specific and dataset-specific configs.
        These configs are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic configs
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save configs

        It will print both current configs and default values(if different).
        It will save configs into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our configs, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        opt.f_map = [opt.crop_size, opt.crop_size * 2, opt.crop_size * 4, opt.crop_size * 8]
        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt