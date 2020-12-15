from matplotlib.animation import FuncAnimation
from IPython.core.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import time
from tqdm import tqdm
import pandas as pd

def tensorboard_vis(summarywriter, step, board_name, img_list, num_row, cmaps, titles, resize=True):
    """
    :param summarywriter: tensorboard summary writer handle
    :param step:
    :param board_name: display name
    :param img_list: a list of images to show
    :param num_row: specify the number of row
    :param cmaps: specify color maps for each image ('gray' for MRI, i.e.)
    :param titles: specify each image title
    :param resize: whether resize the image to show
    :return:
    """
    fig = plt.figure()
    num_figs = len(img_list)
    num_col = np.ceil(num_figs/ num_row)
    print('Visualizing %d images in %d row %d column'%(num_figs, num_row, num_col))

    for i in range(num_figs):
        ax = plt.subplot(num_row, num_col, i + 1)
        if resize:
            tmp = cv2.resize(img_list[i], (256, 256))
        else:
            tmp = img_list[i]
        #print(tmp.shape)
        if isinstance(cmaps, str): c = cmaps
        else: c = cmaps[i]

        if 'seg' in titles[i]:
            vmin = 0
            vmax = 3
        else:
            vmin = -1
            vmax = 1.
        ax.imshow(tmp, cmap=c, vmin=vmin, vmax=vmax), plt.title(titles[i]), plt.axis('off')

    summarywriter.add_figure(board_name, fig, step)
    return summarywriter


def create_group_fig(img_list, num_row, cmaps, titles, resize=True, save_name=None, fig_size=[15,15],
                     vmin=None, vmax=None, is_log=False, dpi=100, format='eps'):
    """
    :param summarywriter: tensorboard summary writer handle
    :param step:
    :param board_name: display name
    :param img_list: a list of images to show
    :param num_row: specify the number of row
    :param cmaps: specify color maps for each image ('gray' for MRI, i.e.)
    :param titles: specify each image title
    :param resize: whether resize the image to show
    :return:
    """
    plt.rcParams['figure.figsize'] = fig_size
    fig = plt.figure()
    num_figs = len(img_list)
    num_col = np.ceil(num_figs / num_row)
    #print('Visualizing %d images in %d row %d column' % (num_figs, num_row, num_col))
    for i in range(num_figs):
        ax = plt.subplot(num_row, num_col, i + 1)
        if resize:
            tmp = cv2.resize(img_list[i], (256, 256))
        else:
            tmp = img_list[i]
        #print(tmp.shape)
        if isinstance(cmaps, str):
            c = cmaps
        else:
            c = cmaps[i]

        if 'seg' in titles[i]:
            v_min, v_max = 0, 3
        else:
            if vmax is not None:
                if isinstance(vmax, list): v_min, v_max = vmin[i], vmax[i]
                else: v_min, v_min = vmin, vmax
            elif tmp.max() <=1:
                v_min, v_max = -1., 1.
            else:
                v_min, v_max = tmp.min(), tmp.max()
        print(v_min, v_max)
        ax.imshow(tmp, cmap=c, vmin=v_min, vmax=v_max), plt.title(titles[i]), plt.axis('off')

    if save_name:
        s = time.time()
        if dpi:
            plt.savefig(save_name,  format=format, dpi=dpi)
        else:
            plt.savefig(save_name, format=format)
        e = time.time()
        #print('save figure %s in %.5f seconds'%(save_name, (e-s)))

    return fig
