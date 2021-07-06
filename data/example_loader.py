import os, sys
sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('./'))
import numpy as np
from glob import glob
import random
import torch
import torch.utils.data
import h5py
from data.augmentation import RandomCrop, Compose, one_hot



class h5lodaer(torch.utils.data.Dataset):
    def __init__(self, is_train, src_key, trg_key, crop_size):
        if is_train and augment:
            self.augment = Compose([
                RandomCrop(self.crop_size)
            ])
        self.src_key = src_key
        self.trg_key = trg_key
        self.crop_size = crop_size

    def read_h5(self, path):
        with h5py.File(path, 'r') as hf:
            self.src_num = hf[src_key]['img'].shape[0]
            self.trg_num = hf[trg_key]['img'].shape[0]

    def __len__(self):
        return max(self.src_num, self.trg_num)

    def __getitem__(self, item):
        with h5py.File(self.path, 'r') as hf:
            src_img, src_seg = hf[self.src_key]['img'][item], hf[self.trg_key]['seg'][item]

            item = np.random.randint(0, self.trg_num)
            trg_img, trg_seg = hf['spectralis']['img'][item], hf['spectralis']['seg'][item]

        xs = np.zeros((1,
                       self.crop_size,
                       self.crop_size)).astype(np.float32)
        ys = np.zeros((1,
                       self.crop_size,
                       self.crop_size)).astype(np.float32)

        xs[0, :, :] = src_img.astype(np.float32).transpose(2,0,1)
        ys[0, :, :] = trg_img.astype(np.float32).transpose(2,0,1)

        dict = {}
        dict['A'] = xs
        dict['B'] = ys

        ss = src_seg.astype(np.uint8)  # w, h
        # One hot encoding
        ss_oh = one_hot(ss)
        dict['A_seg'] = ss_oh

        ss = trg_seg.astype(np.uint8)  # w, h
        ss_oh = one_hot(ss)
        dict['B_seg'] = ss_oh
        return dict
