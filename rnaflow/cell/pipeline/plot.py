import os
from glob import glob
from typing import Optional, Union, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components

import torch
from torch_em.util.util import get_random_colors

from cellpose import plot
from micro_sam.evaluation.model_comparison import _enhance_image

def plot_samples(image: np.ndarray, gt: np.ndarray, 
                segmentation: np.ndarray = None,
                enhance_image: bool = True):
    '''Convenience function to plot images side-by-side

    Source from https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/automatic_segmentation.ipynb
    '''
    n_images = 2 if segmentation is None else 3
    fig, ax = plt.subplots(1, n_images, figsize=(10, 10))

    if enhance_image:
        image = _enhance_image(image, do_norm=True)

    ax[0].imshow(image, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Image")

    gt = connected_components(gt)
    ax[1].imshow(gt, cmap=get_random_colors(gt), interpolation="nearest")
    ax[1].axis("off")
    ax[1].set_title("Ground Truth")

    if n_images == 3:
        ax[2].imshow(segmentation, cmap=get_random_colors(segmentation), interpolation="nearest")
        ax[2].axis("off")
        ax[2].set_title("Prediction")

    plt.show()
    plt.close()