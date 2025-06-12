import os
from abc import ABC, abstractmethod
from pathlib import Path
import glob
from typing import Union, Optional
import importlib

import tifffile as tiff
import numpy as np
import cv2
import torch

from .utils import *
from .preprocess import preprocess_sr
from .track.utils import extract_cell_statistis_from_frames, extract_single_cell_seq_from_track_res
# from .segment import load_detector, load_segmentor, uint16_to_uint8_maxmin, segment_fn, segment_postprocess_fn, map_fn_to_frames, segment_fn, segment_postprocess_fn, \
#                     gray_to_rgb, segment_sr
# from .track import create_csv, track_predict
# from .postprocess import Postprocess, generate_track_csv, extract_track_res

os.environ['CUDA_VISIBLE_DEVICES'] = '7' # TODO: set CUDA device

logger = get_logger(__name__, level=logging.INFO)

class CellPredictor(object):
    """Cell Predictor class for handling cell segmentation and track tasks."""

    def __init__(self,
                stack_path: Path,  # Path to the input cell stack file
                dst_dir: Optional[Path] = None,  # Destination directory for saving results
                seg_with_preprocess: bool = False,  # Whether to use preprocessing for segmentation
                device: str = 'cuda'
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
        # self.gt_data_dir = self.dst_dir / '01_GT'
        self.seg_data_dir = self.dst_dir / '01_SEG'
        self.track_data_dir = self.dst_dir / '01_TRACK'
        self.cell_seq_dir = self.dst_dir / '01_CELL_SEQ'

    def prepare(self, exist_ok: bool = True):
        """Prepare the raw stack by saving it to individual frames."""
        if not self.raw_data_dir.exists() or exist_ok:
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            # Convert the stack to individual frames and save them to the raw data directory.
            logger.info('Saving raw stack to frames in directory: %s', self.raw_data_dir)
            stack_to_frames(self.raw_stack, self.raw_data_dir, prefix='t')

        self.raw_imgs_paths = sorted(self.raw_data_dir.glob('*.tif'))
        del self.raw_stack  # Free memory after saving the stack to frames

    def preprocess(self, exist_ok: bool = True):
        """Preprocess the raw images for segmentation."""
        if not self.raw_data_dir.exists():
            self.pre_data_dir.mkdir(parents=True, exist_ok=True)
        
            logger.info('Begin preprocessing raw images...')
            img0 = tiff.imread(self.raw_imgs_paths[0])
            _ = map_fn_to_frames(self.raw_imgs_paths, preprocess_sr, save_dir=self.pre_data_dir, save_prefix='test_',
                                img_ref=img0)
            
        self.pre_imgs_paths = sorted(self.pre_data_dir.glob('*.tif'))

        # for idx, img_path in enumerate(self.raw_imgs_paths):
        #     img = tiff.imread(img_path)
        #     img = hist_match(img, img0)
        #     img = uint8_to_uint16(img)
        #     dst_img_path = self.pre_data_dir / f'test_{idx:04d}.tif'
        #     tiff.imwrite(dst_img_path, img)

    def segment(self, method: str, exist_ok: bool = True):
        """Segment the preprocessed or raw images using the specified method."""
        if self.seg_with_preprocess:
            input_paths = self.pre_imgs_paths
            logger.info('Using preprocessed images from directory: %s', self.pre_data_dir)
        else:
            input_paths = self.raw_imgs_paths
            logger.info('Using raw images from directory: %s', self.raw_data_dir)
        
        assert method in ['cellpose-sam'], f"Unsupported segmentation method: {method}"
        self.seg_data_dir = self.dst_dir / '01_SEG' / method
        if not self.seg_data_dir.exists() or exist_ok:
            self.seg_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info('Begin segmenting images...')

            if method == 'cellpose-sam':
                module = importlib.import_module('.segment.cellpose_sam_api', package='rnaflow.cell.pipeline')
                segment_fn = getattr(module, 'segment_fn')
                segment_fn(input_paths, self.seg_data_dir, self.device, prefix='seg_')

        
    
        # pre_img0 = tiff.imread(self.pre_imgs_paths[0])
        # min_value0 = np.min(pre_img0)
        # max_value0 = np.max(pre_img0)

        # # convert preprocessed images to RGB format for segmentation
        # logger.info('Converting preprocessed images to RGB format...')
        # self.rgb_data_dir = self.dst_dir / 'PRE' / 'RGBGT'; self.rgb_data_dir.mkdir(parents=True, exist_ok=True)
        # _ = map_fn_to_frames(self.pre_imgs_paths, gray_to_rgb, save_dir=self.rgb_data_dir, save_prefix='test_',
        #                      max_value=max_value0, min_value=min_value0)

        # # for idx, img_path in enumerate(self.pre_imgs_paths):
        # #     img = tiff.imread(img_path)
        # #     if img.dtype == np.uint16:
        # #         img = uint16_to_uint8_maxmin(img, max_value0, min_value0)

        # #     bgr_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # #     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # #     dst_img_path = self.rgb_data_dir / f'test_{idx:04d}.tif'
        # #     tiff.imwrite(dst_img_path, rgb_image)

        # # init segmentation
        # self.rgb_imgs_paths = sorted(glob.glob(str(self.rgb_data_dir / '*.tif')))
        # self.seg_data_dir = self.gt_data_dir / 'SAMSEG'; self.seg_data_dir.mkdir(parents=True, exist_ok=True)
        # detector = load_detector(self.device)
        # sam_predictor = load_segmentor(self.device)
       
        # # segment first frame
        # rgb_img0_path = self.rgb_imgs_paths[0]
        # pre_img0_path = self.pre_imgs_paths[0]
        # mask0, bbox_save = segment_fn(detector, sam_predictor, rgb_img0_path, pre_img0_path)
        # dst_path = self.seg_data_dir / f'man_seg{0:04d}.tif'
        # tiff.imwrite(dst_path, mask0)

        # # segment other frames
        # for frame, img_pre_path, img_rgb_path in enumerate(zip(self.pre_imgs_paths, self.rgb_imgs_paths)):
        #     if frame == 0:
        #         continue
            
        #     # TODO: use the saved bbox from the first frame
        #     mask, _ = segment_fn(detector, sam_predictor, img_rgb_path, img_pre_path, bbox_save)
        #     dst_img_path = self.seg_data_dir / f'man_seg{frame:04d}.tif'
        #     tiff.imwrite(dst_img_path, mask)

        # # segmentation post-processing
        # self.seg_imgs_paths = sorted(glob.glob(str(self.seg_data_dir / '*.tif'))) 
        # self.seg_post_data_dir = self.gt_data_dir / 'SEG'; self.seg_post_data_dir.mkdir(parents=True, exist_ok=True)  # SEG_16

        # for frame, mask_path in enumerate(self.seg_imgs_paths):
        #     mask = tiff.imread(mask_path).astype(np.uint16)
        #     # Apply post-processing to the mask
        #     # TODO: rename - when the mask is empty, it will cause an error
        #     mask_post = segment_postprocess_fn(mask)

        #     dst_img_path = self.seg_post_data_dir / f'man_seg{frame:04d}.tif'
        #     tiff.imwrite(dst_img_path, mask_post)
    
    def track(self, method: str, exist_ok: bool = True):
        """Track the segmented images using the specified method."""
        assert method in ['cell-tracker-gnn', 'trackastra'], f"Unsupported tracking method: {method}"
        self.track_data_dir = self.dst_dir / '01_TRACK' / method
        if not self.track_data_dir.exists() or exist_ok:
            self.track_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info('Begin tracking segmented images...')

            if method == 'cell-tracker-gnn':
                pass
            elif method == 'trackastra':
                module = importlib.import_module('.track.trackastra_api', package='rnaflow.cell.pipeline')
                track_fn = getattr(module, 'cell_track')
                track_fn(self.raw_data_dir, self.seg_data_dir, self.device, self.track_data_dir)



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

    def postprocess(self):
        """Post-process the tracking results to extract cell statistics and sequences."""
        logger.info('Extracting cell statistics from frames...')
        extract_cell_statistis_from_frames(self.raw_data_dir, self.track_data_dir)

        if not self.cell_seq_dir.exists():
            self.cell_seq_dir.mkdir(parents=True, exist_ok=True)
            logger.info('Begin extracting cell sequences from tracking results...')
            # Extract single cell sequences from the tracking results
            extract_single_cell_seq_from_track_res(self.raw_data_dir, self.track_data_dir, self.cell_seq_dir)
        
        logger.info('Post-processing completed. Results saved to: %s', self.cell_seq_dir)
    

        # self.infer_dir = self.dst_dir / '01_RES_inference'

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
        