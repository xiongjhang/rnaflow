import os
from typing import Optional, Union, List
from pathlib import Path
import glob
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import io
from skimage.measure import regionprops

def collect_frames_to_stack(dir: Path, suffix: str = '.tif') -> np.ndarray:
    '''Collects all image files in the specified directory with the given suffix.'''
    assert dir.is_dir(), f"Provided path {dir} is not a directory."
    img_path_list = sorted(dir.glob(f'*{suffix}'))
    imgs = [tiff.imread(img_path) for img_path in img_path_list]
    assert len(imgs) > 1, "There should be more than one image in the directory."

    stack = np.stack(imgs, axis=0)
    assert stack.ndim == 3, "Stack should be a 3D array (time, height, width)."

    return stack

def load_track_res(txt_file_path: Path):
    '''Loads tracking results from a text file, following the Cell Tracking Challenge format.'''
    assert txt_file_path.exists(), f"Tracking result file {txt_file_path} does not exist."
    assert txt_file_path.is_file() and txt_file_path.suffix == '.txt', "Tracking result file should be a .txt file."

    track_res = np.genfromtxt(txt_file_path, dtype=[int, int, int, int])
    return track_res

# MARK: Post-processing functions

def extract_cell_statistis_from_frame(
    img: Union[np.ndarray, Path],
    mask: Union[np.ndarray, Path],
    frame_num: Optional[int] = None,
) -> pd.DataFrame:
    '''Extracts cell statistics from a single frame image and its corresponding mask.'''
    if isinstance(img, Path):
        img = tiff.imread(img)
    if isinstance(mask, Path):
        mask = tiff.imread(mask)
    assert img.ndim == 2 and img.shape == mask.shape, "Input should be 2D array."

    cols = ["seg_label",
            "frame_num",
            "area",
            "bbox_area",
            "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
            "centroid_row", "centroid_col",
            "major_axis_length", "minor_axis_length",
            "max_intensity", "mean_intensity", "min_intensity",
            "orientation", "perimeter",
            "weighted_centroid_row", "weighted_centroid_col"
            ]
    
    num_labels = np.unique(mask).shape[0] - 1
    df = pd.DataFrame(index=range(num_labels), columns=cols)

    for ind, id_res in enumerate(np.unique(mask)):
        # Color 0 is assumed to be background or artifacts
        row_ind = ind - 1
        if id_res == 0:
            continue

        # extracting statistics using regionprops
        properties = regionprops(np.uint8(mask == id_res), img)[0]

        # model feature
        # embedded_feat = self.extract_freature_metric_learning(properties.bbox, img.copy(), result.copy(), id_res)
        # df.loc[row_ind, cols_resnet] = embedded_feat

        # statistics
        df.loc[row_ind, "seg_label"] = id_res

        df.loc[row_ind, "area"], df.loc[row_ind, "bbox_area"] = properties.area, properties.bbox_area

        df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
        df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox

        df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
            properties.centroid[0].round().astype(np.int16), \
            properties.centroid[1].round().astype(np.int16)

        df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
            properties.major_axis_length, properties.minor_axis_length

        df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
            properties.max_intensity, properties.mean_intensity, properties.min_intensity

        df.loc[row_ind, "orientation"], df.loc[row_ind, "perimeter"] = properties.orientation, \
                                                                        properties.perimeter
        if properties.weighted_centroid[0] != properties.weighted_centroid[0] or properties.weighted_centroid[
            1] != properties.weighted_centroid[1]:
            df.loc[row_ind, "weighted_centroid_row"], df.loc[
                row_ind, "weighted_centroid_col"] = properties.centroid
        else:
            df.loc[row_ind, "weighted_centroid_row"], df.loc[
                row_ind, "weighted_centroid_col"] = properties.weighted_centroid
            
    if frame_num is not None:
        df.loc[:, "frame_num"] = int(frame_num)
            
    if df.isnull().values.any():
        warnings.warn("Pay Attention! there are Nan values!")

    return df

def extract_cell_statistis_from_frames(
    img_dir: Path,
    mask_dir: Path,
    save_dir: Optional[Path] = None,
    csv_prefix: str = 'frame_',
):
    """Extracts cell statistics from all frames in the specified image and mask directories."""
    assert img_dir.is_dir() and mask_dir.is_dir(), \
        "Both img_dir and mask_dir must be directories containing image and mask files."
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Image or mask directory does not exist.")
    if save_dir is None:
        save_dir = mask_dir
    else:
        assert save_dir.is_dir(), "save_dir must be a directory."
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob('*.tif'))
    mask_files = sorted(mask_dir.glob('*.tif'))

    assert len(img_files) == len(mask_files), "Number of images and masks must match."

    for frame_num, (img_file, mask_file) in enumerate(zip(img_files, mask_files), 0):
        stats_df = extract_cell_statistis_from_frame(img_file, mask_file, frame_num)
        save_path = save_dir / f"{csv_prefix}{frame_num:04d}.csv"
        stats_df.to_csv(save_path, index=False)

    # all_stats = []
    
    # for img_file, mask_file in zip(img_files, mask_files):
    #     stats = extract_cell_statistis_from_frame(img_file, mask_file)
    #     all_stats.append(stats)

    # df_all_stats = pd.concat(all_stats, ignore_index=True)

    # if save_dir is not None:
    #     if not save_dir.exists():
    #         os.makedirs(save_dir, exist_ok=True)
    #     df_all_stats.to_csv(save_dir / 'cell_statistics.csv', index=False)

    # return df_all_stats

def extract_single_cell_seq_from_track_res(
    raw_img_dir: Path,
    track_res_dir: Path,
    save_dir: Path,
    csv_dir: Optional[Path] = None,
    size: int = 128,
    lazy_load: bool = False,
):
    """Extracts single cell sequences from tracking results and saves them.

    Args:
        raw_img_dir (Path): dir of raw images(*.tif)
        track_res_dir (Path): tracking results directory, including masks(*.tif) and a tracking result(*.txt)
        csv_dir (Path): dir of saving cell statistics
        save_dir (Path): dir of saving single cell sequences
        size (int): height and width of the extracted cell sequence images
        lazy_load (bool): whether to load images and masks lazily, default is False.
    """
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        
    assert raw_img_dir.is_dir() and track_res_dir.is_dir(), \
        "raw_img_dir, track_res_dir, and csv_dir must be directories."
    if lazy_load:
        img_list = sorted(raw_img_dir.glob('*.tif'))
        masks_list = sorted(track_res_dir.glob('*.tif'))
    else:
        img_list = [tiff.imread(img_path) for img_path in sorted(raw_img_dir.glob('*.tif'))]
        masks_list = [tiff.imread(mask_path) for mask_path in sorted(track_res_dir.glob('*.tif'))]
    
    if csv_dir is None:
        csv_dir = track_res_dir
    assert csv_dir.is_dir(), "csv_dir must be a directory."
    csv_list = [pd.read_csv(csv_path) for csv_path in sorted(csv_dir.glob('*.csv'))]
    assert len(img_list) == len(masks_list) == len(csv_list), "Number of images, masks, and CSV files must match."

    txt_file = glob.glob(str(track_res_dir / '*.txt'))
    assert len(txt_file) == 1, "There should be exactly one tracking result file in the directory."
    track_res_path = Path(txt_file[0])
    track_res = load_track_res(track_res_path)

    seq_row = seq_col = size
    num_cells = track_res.shape[0]
    for cell_idx, (cell_id, begin_frame, end_frame, parent_id) in enumerate(track_res, 0):
        # TODO: account for parent_id
        cell_seq_list = []
        empty_cnt = 0
        for frame in range(begin_frame, end_frame + 1):
            if lazy_load:
                cur_img = tiff.imread(img_list[frame])
                cur_mask = tiff.imread(masks_list[frame])
            else:
                cur_img = img_list[frame]
                cur_mask = masks_list[frame]
            cur_csv = csv_list[frame][["seg_label","min_row_bb","min_col_bb","max_row_bb","max_col_bb"]]

            cur_res = np.uint16(cur_mask == cell_id).copy()
            if not np.any(cur_res):
                empty_img = np.zeros((seq_row, seq_col), dtype=np.uint16)
                cell_seq_list.append(empty_img)
                empty_cnt += 1
                continue

            binary_image = np.uint16(cur_res > 0)
            mask_img = np.multiply(cur_img, binary_image) 

            cell_stats = np.array(cur_csv.loc[cur_csv['seg_label'] == cell_id])
            # get the bounding box coordinates
            center_y = (cell_stats[0, 2] + cell_stats[0, 4]) // 2
            center_x = (cell_stats[0, 1] + cell_stats[0, 3]) // 2
            x1 = center_x - seq_col // 2
            y1 = center_y - seq_row // 2
            x2 = center_x + seq_col // 2
            y2 = center_y + seq_row // 2
            # ensure the coordinates are within the image bounds
            pad_top = pad_left = pad_bottom = pad_right = 0
            if x1 < 0:
                pad_top, x1 = abs(x1), 0
            if y1 < 0:
                pad_left, y1 = abs(y1), 0
            if x2 > cur_img.shape[0]:
                pad_bottom, x2 = x2 - cur_img.shape[0], cur_img.shape[0]
            if y2 > cur_img.shape[1]:
                pad_right, y2 = y2 - cur_img.shape[1], cur_img.shape[1]
            # extract the cell sequence image
            cell_img = mask_img[x1:x2, y1:y2]
            # pad the image if it is smaller than the specified size      
            if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                cell_img = np.pad(cell_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                assert cell_img.shape == (seq_row, seq_col), \
                    f"Cell image shape {cell_img.shape} does not match the expected size {(seq_row, seq_col)} after padding."
            cell_seq_list.append(cell_img)

        cell_stack = np.stack(cell_seq_list, axis=0)
        save_path = save_dir / f"cellraw_{cell_id}.tif"
        tiff.imwrite(save_path, cell_stack)