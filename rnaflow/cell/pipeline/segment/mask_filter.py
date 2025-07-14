import os
import os.path as op
from pathlib import Path

import tifffile as tiff
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from skimage import io
from skimage.measure import regionprops
from skimage.morphology import label
import warnings
warnings.filterwarnings("always")

MIN_CELL_SIZE = 2000
MAX_CELL_SIZE = 15000

def get_mask_info(mask: np.ndarray):
    """Get the mask information including the number of cells, cell sizes, and bounding boxes.
    Args:
        mask (np.ndarray): The mask of the cells.
    Returns:
        num_cells (int): The number of cells in the mask.
        cell_labels (np.ndarray): The labels of the cells in the mask.
        cell_sizes (np.ndarray): The sizes of the cells in the mask.
        bounding_boxes (np.ndarray): The bounding boxes of the cells in the mask.
    """
    props = regionprops(mask)
    
    num_cells = len(props)
    cell_labels = np.array([prop.label for prop in props])
    cell_sizes = np.array([prop.area for prop in props])
    bounding_boxes = np.array([prop.bbox for prop in props])
    
    return num_cells, cell_labels, cell_sizes, bounding_boxes


def correct_mask(
        img: np.ndarray,
        mask: np.ndarray,
        save_path: Path = None,
        min_cell_size: int = 2000,
        max_cell_size: int = 15000,
):
    """Post-process the mask to remove small and large cell regions.
    
    Adapted from https://github.com/talbenha/cell-tracker-gnn.
    """
    per_cell_change = False
    per_mask_change = False
    res_save = mask.copy()
    labels_mask = mask.copy()
    
    # Filter cell regions based on size
    while True:
        bin_mask = labels_mask > 0
        re_label_mask = label(bin_mask, connectivity=1)
        un_labels, counts = np.unique(re_label_mask, return_counts=True)

        if min_cell_size and np.any(counts < min_cell_size):
            per_mask_change = True
            first_label_ind = np.argwhere(counts < min_cell_size)
            if first_label_ind.size > 1:
                first_label_ind = first_label_ind.squeeze()[0]
            first_label_num = un_labels[first_label_ind]
            labels_mask[re_label_mask == first_label_num] = 0
        elif max_cell_size and np.any(counts > max_cell_size):
            per_mask_change = True
            first_label_ind = np.argwhere(counts > max_cell_size)
            if first_label_ind.size > 1:
                first_label_ind = first_label_ind.squeeze()[0]
            first_label_num = un_labels[first_label_ind]
            labels_mask[re_label_mask == first_label_num] = 0
        else:
            break
    
    bin_mask = (labels_mask > 0) * 1.0
    result = np.multiply(result, bin_mask)
    if not np.all(np.unique(result) == np.unique(res_save)):
        warnings.warn(
            f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")
    
    # Iterate through each unique label in the mask
    for ind, id_res in enumerate(np.unique(result)):
        if id_res == 0:
            continue
        bin_mask = (result == id_res).copy() # binary mask for the current label
        # remove small and large regions iteratively
        while True:
            re_label_mask = label(bin_mask)
            un_labels, counts = np.unique(re_label_mask, return_counts=True)

            if np.any(counts < min_cell_size):
                per_cell_change = True
                # print(f"{im_path}: \n {counts}")
                first_label_ind = np.argwhere(counts < min_cell_size)
                if first_label_ind.size > 1:
                    first_label_ind = first_label_ind.squeeze()[0]
                first_label_num = un_labels[first_label_ind]
                curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                bin_mask[curr_mask] = False
                result[curr_mask] = 0.0
            elif np.any(counts > max_cell_size):
                per_cell_change = True
                # print(f"{im_path}: \n {counts}")
                first_label_ind = np.argwhere(counts > max_cell_size)
                if first_label_ind.size > 1:
                    first_label_ind = first_label_ind.squeeze()[0]
                first_label_num = un_labels[first_label_ind]
                curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                bin_mask[curr_mask] = False
                result[curr_mask] = 0.0
            else:
                break

        # If there are still multiple labels in the binary mask, choose the largest one
        while True:
            re_label_mask = label(bin_mask)
            un_labels, counts = np.unique(re_label_mask, return_counts=True)
            if un_labels.shape[0] > 2:
                per_cell_change = True
                # n_changes += 1
                # print(f"un_labels.shape[0] > 2 : {im_path}: \n {counts}")
                first_label_ind = np.argmin(counts)
                if first_label_ind.size > 1:
                    first_label_ind = first_label_ind.squeeze()[0]
                first_label_num = un_labels[first_label_ind]
                curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                bin_mask[curr_mask] = False
                result[curr_mask] = 0.0
            else:
                break

    if not np.all(np.unique(result) == np.unique(res_save)):
        warnings.warn(
            f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")
        
    if per_cell_change or per_mask_change:
        res1 = (res_save > 0) * 1.0
        res2 = (result > 0) * 1.0
        n_pixels = np.abs(res1 - res2).sum()
        print(f"per_mask_change={per_mask_change}, per_cell_change={per_cell_change}, number of changed pixels: {n_pixels}")
    
    if save_path is not None:
        tiff.imwrite(save_path, result.astype(np.uint16))
    
    return result.astype(np.uint16)

def correct_mask_fast(
        img: np.ndarray,
        mask: np.ndarray,
        save_path: Path = None,
        min_cell_size: int = 500,
        max_cell_size: int = 16000,
):
    """Fast version of the mask correction."""
    per_cell_change = False
    per_mask_change = False
    res_save = mask.copy()
    labels_mask = mask.copy()

    '''Time Counsumption(cellpose-sam):
    1. img size=(4666, 2023), bs=256, ~14s/frame
    2. img size=(7016, 2048), bs=256, ~16s/frame
    '''
    # Filter cell regions based on size
    re_label_mask = label(labels_mask, connectivity=2)
    num_cells, cell_labels, cell_sizes, bounding_boxes = get_mask_info(re_label_mask)
    
    index_to_remove = np.where((cell_sizes < min_cell_size) | (cell_sizes > max_cell_size))[0]
    if index_to_remove.size > 0:
        per_mask_change = True
        for index in index_to_remove:
            labels_mask[re_label_mask == cell_labels[index]] = 0

    bin_mask = (labels_mask > 0) * 1.0
    result = np.multiply(mask, bin_mask)
    # if not np.all(np.unique(result) == np.unique(res_save)):
    #     warnings.warn(
    #         f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")

    '''Time Counsumption(cellpose-sam):
    '''
    # If there are still multiple labels in the binary mask, choose the largest one
    # for ind, id_res in enumerate(np.unique(result)):
    #     if id_res == 0:
    #         continue
    #     bin_mask = (result == id_res).copy()
    #     re_label_mask = label(bin_mask, connectivity=1)
    #     un_labels, counts = np.unique(re_label_mask, return_counts=True)
    #     if un_labels.shape[0] > 2:
    #         per_cell_change = True
    #         # n_changes += 1
    #         # print(f"un_labels.shape[0] > 2 : {im_path}: \n {counts}")
    #         first_label_ind = np.argmin(counts)
    #         if first_label_ind.size > 1:
    #             first_label_ind = first_label_ind.squeeze()[0]
    #         first_label_num = un_labels[first_label_ind]
    #         curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
    #         bin_mask[curr_mask] = False
    #         result[curr_mask] = 0.0

    if per_cell_change or per_mask_change:
        res1 = (res_save > 0) * 1.0
        res2 = (result > 0) * 1.0
        n_pixels = np.abs(res1 - res2).sum()
        # print(f"per_mask_change={per_mask_change}, per_cell_change={per_cell_change}, number of changed pixels: {n_pixels}")

    if save_path is not None:
        tiff.imwrite(save_path, result.astype(np.uint16))

    return result.astype(np.uint16)
