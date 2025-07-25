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

from rnaflow.cell.utils import *
from rnaflow.cell.pipeline.preprocess import preprocess_sr
from rnaflow.cell.pipeline.track.utils import extract_cell_statistis_from_frames, extract_single_cell_seq_from_track_res
# from .track import create_csv, track_predict
# from .postprocess import Postprocess, generate_track_csv, extract_track_res

logger = get_logger(__name__, level=logging.INFO)

class CellPredictor(object):
    """Cell Predictor class for handling cell segmentation and track tasks."""

    def __init__(
            self,
            stack_path: Path,  # Path to the input cell stack file
            dst_dir: Optional[Path] = None,  # Destination directory for saving results
            device: str = 'cuda'
    ):
        super().__init__()

        # prepare directories and paths
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
        self.device = device
        
        logger.info('Reading cell stack from: %s', self.raw_stack_path)
        self.raw_stack = tiff.imread(self.raw_stack_path)
        assert self.raw_stack.ndim == 3, f"Input stack must be a 3D array, got shape {self.raw_stack.shape}"
        self.num_frames, self.height, self.width = self.raw_stack.shape
        self.dtype = self.raw_stack.dtype
        
        self.raw_data_dir = self.dst_dir / '01'
        self.pre_data_dir = self.dst_dir / 'PRE' / 'PRE_MUL'
        # self.gt_data_dir = self.dst_dir / '01_GT'
        self.seg_data_dir_raw = self.dst_dir / '01_SEG'
        self.seg_data_dir_curated = self.dst_dir / '01_SEG_curated'
        self.track_data_dir = self.dst_dir / '01_TRACK'
        self.cell_seq_dir = self.dst_dir / '01_CELL_SEQ'

        # exist_ok = True - if the output directory already exists, it will be overwritten.
        # exist_ok = False - if the output directory already exists, it will not be overwritten.
        self.prepare(exist_ok=False) 

    def prepare(self, exist_ok: bool = False):
        """Prepare the raw stack by saving it to individual frames."""
        if not self.raw_data_dir.exists() or exist_ok:
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            # Convert the stack to individual frames and save them to the raw data directory.
            logger.info('Saving raw stack to frames in directory: %s', self.raw_data_dir)
            stack_to_frames(self.raw_stack, self.raw_data_dir, prefix='t')
            logger.info('Raw stack saved successfully. Number of frames: %d', self.num_frames)

        self.raw_imgs_paths = sorted(self.raw_data_dir.glob('*.tif'))
        del self.raw_stack  # Free memory after saving the stack to frames


    def preprocess(self, exist_ok: bool = False):
        """Preprocess the raw images for segmentation."""
        if not self.pre_data_dir.exists() or exist_ok:
            self.pre_data_dir.mkdir(parents=True, exist_ok=True)
        
            logger.info('Begin preprocessing raw images...')
            img0 = tiff.imread(self.raw_imgs_paths[0])
            _ = map_fn_to_frames(self.raw_imgs_paths, preprocess_sr, save_dir=self.pre_data_dir, save_prefix='test',
                                img_ref=img0)
            
        self.pre_imgs_paths = sorted(self.pre_data_dir.glob('*.tif'))

    # region Segmentation

    @staticmethod
    def segment_static(
            input_paths: Union[Path, list],  # List of input image paths or a single path
            method: str, 
            exist_ok: bool = True,
            seg_data_dir_raw: Optional[Path] = None,
            seg_data_dir_curated: Optional[Path] = None,
            **kwargs
    ): 
        """Static method to segment images using the specified method."""
        assert method in ['cellpose-sam', 'sunrui-version'], f"Unsupported segmentation method: {method}"

        if not seg_data_dir_raw.exists() or exist_ok:
            seg_data_dir_raw.mkdir(parents=True, exist_ok=True)
            seg_data_dir_curated.mkdir(parents=True, exist_ok=True)
            logger.info('Begin segmenting images...')

            if method == 'cellpose-sam':
                module = importlib.import_module('.segment.cellpose_sam_api', package='rnaflow.cell.pipeline')
                segment_fn = getattr(module, 'segment_fn')
                segment_fn(
                    input_paths,
                    seg_data_dir_raw,
                    seg_data_dir_curated,
                    prefix='seg_'
                )

            elif method == 'sunrui-version':
                module = importlib.import_module('.segment.sam_sr_api', package='rnaflow.cell.pipeline')
                segment_fn = getattr(module, 'segment_fn')
                segment_fn(
                    input_paths,
                    seg_data_dir_raw,
                    seg_data_dir_curated,
                    prefix='seg_',
                    **kwargs
                )

            logger.info('Segmentation completed. Results saved to: %s', seg_data_dir_raw)

    def segment(
            self,
            method: str,
            seg_with_preprocess: bool = False,  # Whether to use preprocessing for segmentation
            exist_ok: bool = True,
    ):
        """Segment the preprocessed or raw images using the specified method."""
        if seg_with_preprocess:
            input_paths = self.pre_imgs_paths
            logger.info('Using preprocessed images from directory: %s', self.pre_data_dir)
        else:
            input_paths = self.raw_imgs_paths
            logger.info('Using raw images from directory: %s', self.raw_data_dir)

        self.seg_data_dir_raw = self.seg_data_dir_raw / method
        self.seg_data_dir_curated = self.seg_data_dir_curated / method
        
        self.segment_static(
            input_paths=input_paths,
            method=method,
            exist_ok=exist_ok,
            seg_data_dir_raw=self.seg_data_dir_raw,
            seg_data_dir_curated=self.seg_data_dir_curated
        )

    # region Tracking

    @staticmethod
    def track_static(
            method: str,  # Tracking method to use
            raw_data_dir: Path,
            mask_data_dir: Path,
            exist_ok: bool = True,
            device: str = 'cuda',
            track_data_dir: Optional[Path] = None,
            extract_cell_info: bool = True,
            **kwargs
    ):
        """Static method to track segmented images using the specified method."""
        assert method in ['cell-tracker-gnn', 'trackastra'], f"Unsupported tracking method: {method}"

        if not track_data_dir.exists() or exist_ok:
            track_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info('Begin tracking segmented images...')

            if method == 'cell-tracker-gnn':
                pass
            elif method == 'trackastra':
                module = importlib.import_module('.track.trackastra_api', package='rnaflow.cell.pipeline')
                track_fn = getattr(module, 'cell_track')
                track_fn(raw_data_dir, mask_data_dir, device, track_data_dir)

            logger.info('Tracking completed. Results saved to: %s', track_data_dir)

            if extract_cell_info:
                # Extract cell statistics from the tracking results
                logger.info('Extracting cell statistics from frames...')
                extract_cell_statistis_from_frames(raw_data_dir, track_data_dir)
                logger.info('Cell statistics extraction completed. Results saved to: %s', track_data_dir)

    def track(
            self, 
            method: str, 
            exist_ok: bool = True
    ):
        """Track the segmented images using the specified method."""
        
        self.track_data_dir = self.track_data_dir / method

        self.track_static(
            method=method,
            raw_data_dir=self.raw_data_dir,
            mask_data_dir=self.seg_data_dir_curated,
            exist_ok=exist_ok,
            device=self.device,
            track_data_dir=self.track_data_dir
        )

    # region Post-processing

    def extract_cell_seq(self, exist_ok: bool = True):
        """Post-process the tracking results to extract cell statistics and sequences."""
        if not self.cell_seq_dir.exists() or exist_ok:
            self.cell_seq_dir.mkdir(parents=True, exist_ok=True)
            logger.info('Begin extracting cell sequences from tracking results...')
            # Extract single cell sequences from the tracking results
            extract_single_cell_seq_from_track_res(self.raw_data_dir, self.track_data_dir, self.cell_seq_dir)
        
        logger.info('Post-processing completed. Results saved to: %s', self.cell_seq_dir)
        