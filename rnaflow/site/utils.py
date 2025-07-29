import os, warnings, glob, shutil
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from tqdm import tqdm
from pathlib import Path

from cellpose.io import imread

def get_image_files(folder, mask_filter, imf=None, recursive=False):
    """
    Finds all images in a folder and its subfolders (if specified) with the given file extensions.

    Args:
        folder (str): The path to the folder to search for images.
        mask_filter (str): The filter for mask files.
        imf (str, optional): The additional filter for image files. Defaults to None.
        recursive (bool, optional): Whether to search recursively in subfolders. Defaults to False.

    Returns:
        list: A list of image file paths.

    Raises:
        ValueError: If no files are found in the specified folder.
        ValueError: If no images are found in the specified folder with the supported file extensions.
        ValueError: If no images are found in the specified folder without the mask or flow file endings.
    """
    mask_filters = ["_cp_output", "_flows", "_flows_0", "_flows_1",
                    "_flows_2", "_cellprob", "_masks", mask_filter]
    image_names = []
    if imf is None:
        imf = ""

    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".flex", ".dax", ".nd2", ".nrrd"]
    ext_patterns = [f"*{imf}{ext}" for ext in exts] + [f"*{imf}{ext.upper()}" for ext in exts]

    folders = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            folders.append(root)

    l0 = 0
    al = 0
    for folder in folders:
        all_files = glob.glob(folder + "/*")
        al += len(all_files)
        for ext_pattern in ext_patterns:
            image_names.extend(glob.glob(os.path.join(folder, ext_pattern)))
        l0 += len(image_names)

    # return error if no files found
    if al == 0:
        raise ValueError("ERROR: no files in --dir folder ")
    elif l0 == 0:
        raise ValueError(
            "ERROR: no images in --dir folder with extensions .png, .jpg, .jpeg, .tif, .tiff, .flex"
        )

    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and
                      imfile[-len(mask_filter):] != mask_filter) or
                     len(imfile) <= len(mask_filter) for mask_filter in mask_filters])
        if len(imf) > 0:
            igood &= imfile[-len(imf):] == imf
        if igood:
            imn.append(im)

    image_names = imn

    # remove duplicates
    image_names = [*set(image_names)]
    image_names = natsorted(image_names)

    if len(image_names) == 0:
        raise ValueError(
            "ERROR: no images in --dir folder without _masks or _flows or _cellprob ending")

    return image_names

def get_label_files(image_names, mask_filter, imf=None):
    """
    Get the label files corresponding to the given image names and mask filter.

    Args:
        image_names (list): List of image names.
        mask_filter (str): Mask filter to be applied.
        imf (str, optional): Image file extension. Defaults to None.

    Returns:
        tuple: A tuple containing the label file names and flow file names (if present).
    """
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    if os.path.exists(label_names[0] + mask_filter + ".tif"):
        label_names = [label_names[n] + mask_filter + ".tif" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".tiff"):
        label_names = [label_names[n] + mask_filter + ".tiff" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".png"):
        label_names = [label_names[n] + mask_filter + ".png" for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + ".nrrd"):
        label_names = [label_names[n] + mask_filter + ".nrrd" for n in range(nimg)]
    else:
        label_names = None

    if not all([os.path.exists(label) for label in label_names]):
        label_names = None

    return label_names

def load_images_labels(tdir, mask_filter="_masks", image_filter=None,
                       recusive=False, read_data: bool = True):
    """
    Loads images and corresponding labels from a directory.

    Args:
        tdir (str): The directory path.
        mask_filter (str, optional): The filter for mask files. Defaults to "_masks".
        image_filter (str, optional): The filter for image files. Defaults to None.
        recusive (bool, optional): Whether to search recursively in subfolders. Defaults to False.
        read_data (bool, optional): Whether to read the data from the files. Defaults to True.

    Returns:
        tuple: A tuple containing a list of images, a list of labels, and a list of image names.
    """
    image_names = get_image_files(tdir, mask_filter, image_filter, recusive)
    nimg = len(image_names)

    # training data
    label_names = get_label_files(image_names, mask_filter,
                                              imf=image_filter)
    
    images = []
    labels = []
    if read_data:
        k = 0
        for n in range(nimg):
            if os.path.isfile(label_names[n]):
                image = imread(image_names[n])
                if label_names is not None:
                    label = imread(label_names[n])
                images.append(image)
                labels.append(label)
                k += 1
    return images, labels, image_names