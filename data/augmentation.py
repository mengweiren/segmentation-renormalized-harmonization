
import numpy as np
import cv2
import random


def one_hot(seg, ncols = 4):
    ss_oh = np.zeros((seg.size, ncols), dtype=np.float32)
    ss_oh[np.arange(seg.size), seg.ravel()] = 1
    ss_oh = np.reshape(ss_oh, seg.shape + (ncols,))
    return ss_oh.transpose(2, 0, 1)


class RandomCrop(object):
    def __init__(self, crop_size=128, pre_center=False, range=()):
        self.crop_size = crop_size
        self.max_trail = 50
        self.pre_crop = pre_center
        if pre_center:
            self.xmin, self.xmax, self.ymin, self.ymax = range

    def __call__(self, image, masks):
        if self.pre_crop:
            image = image[self.xmin:self.xmax, self.ymin:self.ymax]
            masks = masks[self.xmin:self.xmax, self.ymin:self.ymax]
        x, y, c = image.shape
        sx, sy, sc = masks.shape
        imgout = np.zeros([self.crop_size, self.crop_size, c])
        if image.shape == masks.shape:
            prob_seg = False
            maskout = np.zeros([self.crop_size, self.crop_size])
        else:
            prob_seg = True
            maskout = np.zeros([self.crop_size, self.crop_size, sc])
        rand_range_x = x - self.crop_size
        rand_range_y = y - self.crop_size
        if rand_range_x <= 0:
            x_offset = 0
        else:
            x_offset = np.random.randint(rand_range_x)#rand_range_x = 0
        if rand_range_y <= 0:
            y_offset = 0
        else:
            y_offset = np.random.randint(rand_range_y)
        tmp = image[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size]
        i = 0
        while np.sum(tmp) == 0. and i < self.max_trail and rand_range_x > 0:
            x_offset = np.random.randint(rand_range_x)
            y_offset = np.random.randint(rand_range_y)
            tmp = image[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size]
            i += 1
        imgout[:x, :y] = tmp
        if prob_seg:
            maskout[:x, :y] = masks[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size, :]
        else:
            maskout[:x, :y] = masks[x_offset: x_offset + self.crop_size, y_offset: y_offset + self.crop_size, 0]
        return imgout, maskout


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels
