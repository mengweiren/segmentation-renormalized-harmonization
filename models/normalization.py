import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision.models as models


class SegFeature(nn.Module):
    def __init__(self, label_nc, ngf=64, feat_nc=512):
        super().__init__()
        self.convfeat = nn.Sequential(
            nn.Conv2d(label_nc, ngf, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(ngf, 2*ngf, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(4*ngf, 8*ngf, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True))
        self.avp = nn.AdaptiveAvgPool2d((7, 7))
        self.mlp = nn.Sequential(
            nn.Linear(8*ngf * 7 * 7, 8*ngf),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(8*ngf, feat_nc))

    def forward(self, image, seg_map):
        seg_feat = self.convfeat(seg_map)
        seg_feat = self.avp(seg_feat)
        seg_feat = torch.flatten(seg_feat, 1)
        seg_feat = self.mlp(seg_feat)
        return seg_feat



# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    """
    Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
    Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
    """
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class SharedSRenorm(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc=4):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = 3
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        nhidden = 512
        self.mlp_gamma = nn.Linear(nhidden, norm_nc)
        self.mlp_beta = nn.Linear(nhidden, norm_nc)

    def forward(self, x, seg_feat):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        gamma = self.mlp_gamma(seg_feat)
        b, c = gamma.size()
        gamma = gamma.view(b, c, 1, 1)

        beta = self.mlp_beta(seg_feat)
        beta = beta.view(b, c, 1, 1)

        # apply scale and bias
        out = normalized * (1 + gamma.expand_as(normalized)) + beta.expand_as(normalized)

        return out