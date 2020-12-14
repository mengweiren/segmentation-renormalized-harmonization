
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

'''Geometric augmentation on image+masks'''
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        seg (Image): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, seg)
        img (Image): the cropped image
        seg (Image): the cropped segmentation
    """
    def __init__(self, pre_center=False, range=()):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.pre_crop = pre_center
        if pre_center:
            self.xmin, self.xmax, self.ymin, self.ymax = range

    def __call__(self, image, masks):
        if self.pre_crop:
            image = image[self.xmin:self.xmax, self.ymin:self.ymax]
            masks = masks[self.xmin:self.xmax, self.ymin:self.ymax]
        height, width, _ = image.shape
        mheight, mwidth, _ = masks.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, masks

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                current_mask = masks

                w = random.uniform(0.3 * width, width)
                #h = w
                h = random.uniform(0.3 * height, height)
                mw = w * (mwidth/ width)
                mh = h * (mheight/ height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2 or mh / mw < 0.5 or mh / mw > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)
                mleft = left * (mwidth/ width)
                mtop = top* (mheight/ height)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # convert to integer rect x1,y1,x2,y2
                mrect = np.array([int(mleft), int(mtop), int(mleft + mw), int(mtop + mh)])

                # cut the crop from the image
                current_mask = current_mask[mrect[1]:mrect[3], mrect[0]:mrect[2], :]
                if np.sum(current_mask) == 0:
                    continue
                #current_image = np.reshape(current_image, current_image.shape + (1,))
                #current_mask = np.reshape(current_mask, current_mask.shape + (1,))
                #print(current_image.shape, current_mask.shape)
                return current_image, current_mask

class Resize_all(object):
    def __init__(self, size=(128, 128)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.height, self.width = self.size
        self.mheight, self.mwidth = self.size

    def __call__(self, image, masks):
        image = cv2.resize(image, (self.width, self.height))
        masks = cv2.resize(masks, (self.mwidth, self.mheight))
        image = np.reshape(image, image.shape + (1,))
        #masks = np.reshape(masks, masks.shape + (1,))
        #print(image.shape)
        return image, masks


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
