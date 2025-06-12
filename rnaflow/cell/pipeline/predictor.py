import os
from abc import ABC, abstractmethod
from pathlib import Path
import glob
from typing import Union, Optional

import tifffile as tiff
import numpy as np
import cv2
import torch

from .utils import *
from .preprocess import preprocess_sr
from .segment import load_detector, load_segmentor, uint16_to_uint8_maxmin, segment_fn, segment_postprocess_fn, map_fn_to_frames, segment_fn, segment_postprocess_fn, \
                    gray_to_rgb, segment_sr
# from .track import create_csv, track_predict
# from .postprocess import Postprocess, generate_track_csv, extract_track_res

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: set CUDA device

logger = get_logger(__name__, level=logging.INFO)

class CellPredictor(object):
    """Cell Predictor class for handling cell segmentation and track tasks."""

    def __init__(self,
                stack_path: Path,  # Path to the input cell stack file
                dst_dir: Optional[Path] = None,  # Destination directory for saving results
                seg_with_preprocess: bool = False,  # Whether to use preprocessing for segmentation
                device: str = 'cuda: 0'
            ):
        super().__init__()

        self.stack_stem = stack_path.stem  # cell stack name without extension
        self.raw_stack_path = stack_path
        if dst_dir is None:
            # If dst_dir is not provided, use the parent directory of the stack path
            self.dst_dir = stack_path.parent / self.stack_stem
        else:
            self.dst_dir = dst_dir
        if not self.dst_dir.exists():
            self.dst_dir.mkdir(parents=True, exist_ok=True)

        # global parameters
        self.seg_with_preprocess = seg_with_preprocess
        self.device = device
        
        logger.info('Reading cell stack from: %s', self.raw_stack_path)
        self.raw_stack = tiff.imread(self.raw_stack_path)
        assert self.raw_stack.ndim == 3, f"Input stack must be a 3D array, got shape {self.raw_stack.shape}"
        self.num_frames, self.height, self.width = self.raw_stack.shape
        self.dtype = self.raw_stack.dtype
        
        self.raw_data_dir = self.dst_dir / '01'
        self.pre_data_dir = self.dst_dir / 'PRE' / 'PRE_MUL'
        self.gt_data_dir = self.dst_dir / '01_GT'
        self.res_data_dir = self.dst_dir / '01_RES'

    def prepare(self):
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.gt_data_dir.mkdir(parents=True, exist_ok=True)
        # Convert the stack to individual frames and save them to the raw data directory.
        logger.info('Saving raw stack to frames in directory: %s', self.raw_data_dir)
        stack_to_frames(self.raw_stack, self.raw_data_dir, prefix='t')

        del self.raw_stack  # Free memory after saving the stack to frames

    def preprocess(self):
        self.pre_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_imgs_paths = sorted(glob.glob(str(self.raw_data_dir / '*.tif')))

        logger.info('Begin preprocessing raw images...')
        img0 = tiff.imread(self.raw_imgs_paths[0])
        _ = map_fn_to_frames(self.raw_imgs_paths, preprocess_sr, save_dir=self.pre_data_dir, save_prefix='test_',
                             img_ref=img0)

        # for idx, img_path in enumerate(self.raw_imgs_paths):
        #     img = tiff.imread(img_path)
        #     img = hist_match(img, img0)
        #     img = uint8_to_uint16(img)
        #     dst_img_path = self.pre_data_dir / f'test_{idx:04d}.tif'
        #     tiff.imwrite(dst_img_path, img)

    def segment(self):
        self.pre_imgs_paths = sorted(glob.glob(str(self.pre_data_dir / '*.tif')))
        logger.info('Begin segmenting images...')
        pre_img0 = tiff.imread(self.pre_imgs_paths[0])
        min_value0 = np.min(pre_img0)
        max_value0 = np.max(pre_img0)

        # convert preprocessed images to RGB format for segmentation
        logger.info('Converting preprocessed images to RGB format...')
        self.rgb_data_dir = self.dst_dir / 'PRE' / 'RGBGT'; self.rgb_data_dir.mkdir(parents=True, exist_ok=True)
        _ = map_fn_to_frames(self.pre_imgs_paths, gray_to_rgb, save_dir=self.rgb_data_dir, save_prefix='test_',
                             max_value=max_value0, min_value=min_value0)

        # for idx, img_path in enumerate(self.pre_imgs_paths):
        #     img = tiff.imread(img_path)
        #     if img.dtype == np.uint16:
        #         img = uint16_to_uint8_maxmin(img, max_value0, min_value0)

        #     bgr_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #     dst_img_path = self.rgb_data_dir / f'test_{idx:04d}.tif'
        #     tiff.imwrite(dst_img_path, rgb_image)

        # init segmentation
        self.rgb_imgs_paths = sorted(glob.glob(str(self.rgb_data_dir / '*.tif')))
        self.seg_data_dir = self.gt_data_dir / 'SAMSEG'; self.seg_data_dir.mkdir(parents=True, exist_ok=True)
        detector = load_detector(self.device)
        sam_predictor = load_segmentor(self.device)
       
        # segment first frame
        rgb_img0_path = self.rgb_imgs_paths[0]
        pre_img0_path = self.pre_imgs_paths[0]
        mask0, bbox_save = segment_fn(detector, sam_predictor, rgb_img0_path, pre_img0_path)
        dst_path = self.seg_data_dir / f'man_seg{0:04d}.tif'
        tiff.imwrite(dst_path, mask0)

        # segment other frames
        for frame, img_pre_path, img_rgb_path in enumerate(zip(self.pre_imgs_paths, self.rgb_imgs_paths)):
            if frame == 0:
                continue
            
            # TODO: use the saved bbox from the first frame
            mask, _ = segment_fn(detector, sam_predictor, img_rgb_path, img_pre_path, bbox_save)
            dst_img_path = self.seg_data_dir / f'man_seg{frame:04d}.tif'
            tiff.imwrite(dst_img_path, mask)

        # segmentation post-processing
        self.seg_imgs_paths = sorted(glob.glob(str(self.seg_data_dir / '*.tif'))) 
        self.seg_post_data_dir = self.gt_data_dir / 'SEG'; self.seg_post_data_dir.mkdir(parents=True, exist_ok=True)  # SEG_16

        for frame, mask_path in enumerate(self.seg_imgs_paths):
            mask = tiff.imread(mask_path).astype(np.uint16)
            # Apply post-processing to the mask
            # TODO: rename - when the mask is empty, it will cause an error
            mask_post = segment_postprocess_fn(mask)

            dst_img_path = self.seg_post_data_dir / f'man_seg{frame:04d}.tif'
            tiff.imwrite(dst_img_path, mask_post)
    
    # def track(self):
    #     self.seg_post_imgs_paths = sorted(glob.glob(str(self.seg_post_data_dir / '*.tif')))
    #     self.track_csv_dir = self.dst_dir / '01_CSV'; self.track_csv_dir.mkdir(parents=True, exist_ok=True)
        
    #     # preprocess_seq2graph_clean - 01_CSV
    #     min_cell_size = 20
    #     input_model = '/root/cell_track/bash/cell-tracker-gnn-main/outputs/2023-11-20/13-48-43/all_params.pth'
    #     create_csv(self.raw_data_dir, self.seg_post_data_dir, input_model, self.track_csv_dir, min_cell_size)

    #     # inference_clean - 01_RES_inference
    #     model_path = r'/root/cell_track/bash/cell-tracker-gnn-main/logs/runs/2023-11-20/21-09-37/checkpoints/epoch=274.ckpt'
    #     num_seq = r"01"
    #     assert num_seq == '01' or num_seq == '02'
    #     track_predict(model_path, self.dst_dir, num_seq)

    # def postprocess(self):
    #     self.infer_dir = self.dst_dir / '01_RES_inference'

    #     # postprocess_clean
    #     modality = '2D'
    #     is_3d = '3d' in modality.lower() 
    #     directed = True
    #     merge_operation = 'AND'
    #     pp = Postprocess(is_3d=is_3d,
    #                     type_masks='tif', merge_operation=merge_operation,
    #                     decision_threshold=0.5,
    #                     path_inference_output=self.infer_dir, center_coord=False,
    #                     directed=directed,
    #                     path_seg_result=self.seg_post_data_dir)
        
    #     all_frames_traject, trajectory_same_label, df_trajectory, str_track = pp.create_trajectory()
    #     pp.fill_mask_labels(debug=False)

    #     # single fast large
    #     self.track_res_dir = self.dst_dir / '01_GT' / '_RES'
    #     mask_track_imgs_paths = sorted(glob.glob(str(self.track_res_dir / '*.tif')))
    #     mask_track_imgs = [tiff.imread(img_path) for img_path in mask_track_imgs_paths]
    #     mask_track_stack = np.stack(mask_track_imgs, axis=0)
    #     self.raw_stack = tiff.imread(self.raw_stack_path)
    #     generate_track_csv(self.raw_stack, mask_track_stack, self.track_res_dir)

    #     track_res_path = self.track_res_dir / 'res_track.txt'
    #     track_res = np.genfromtxt(track_res_path, dtype=[int, int, int, int])
    #     self.track_seq_dir = self.dst_dir / '01_GT' / 'maskOriginal';  self.track_seq_dir.mkdir(parents=True, exist_ok=True)
    #     extract_track_res(self.raw_stack, mask_track_stack, track_res, self.dst_dir / '01_GT')
        