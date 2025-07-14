from typing import Any, Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import logging

import numpy as np

from cellpose import models
from cellpose.io import imread, imsave

from rnaflow.cell.pipeline.utils import get_logger
from rnaflow.cell.pipeline.segment.mask_filter import correct_mask_fast

logger = get_logger(__name__, level=logging.INFO)

# Prepare for Cellpose model
def prepare(gpu=True):
    logger.info(f"Loading Cellpose model...")
    model = models.CellposeModel(gpu=gpu)
    return model

# Pure inference function for segmenting images, should support both single and batch inference
def seg_per_frame(
        segmentor, 
        img: Union[Path, List[Path], np.ndarray],
        dst: Optional[Union[Path, List[Path]]] = None,
        batch_size: int = 64,
        niter: int = 1000,
) -> List[np.ndarray]:
    '''Cellpose-sam inference function.

    If `img` is a single image (Path or np.ndarray), it will return the mask for that image.
    If `img` is a list of images (List[Path]), it will return a batch of masks.

    '''
    if isinstance(img, Path):
        input = [imread(str(img))]
    elif isinstance(img, np.ndarray):
        input = [img]
    elif isinstance(img, list):
        input = [imread(p) for p in img]
    else:
        raise ValueError(f"Unsupported input type: {type(img)}")

    # input can be list of 2D/3D/4D images, or array of 2D/3D/4D images.
    masks_pred, flows, styles = segmentor.eval(
                                    input, 
                                    batch_size=batch_size, 
                                    niter=niter
                                )
    if isinstance(dst, Path):
        imsave(str(dst), masks_pred[0].astype(np.uint16))
    elif isinstance(dst, list):
        for i, mask in enumerate(masks_pred):
            imsave(str(dst[i]), mask.astype(np.uint16))

    return masks_pred

# Pipeline function to segment images and save results
def segment_fn(
        input: Union[Path, List[Path]],
        save_raw_dir: Path,
        save_curated_dir: Path,
        prefix: str = 'frame_',
        batch_size: int = 256,
):
    """Segment an image using the Cellpose model."""
    # TODO: support mask filter
    # TODO: support batch inference

    if isinstance(input, Path):
        input =  sorted(input.glob('*.tif'))
    img_paths = input
    
    # get model
    segmentor = prepare()
    
    # model inference
    for idx, img_path in tqdm(enumerate(img_paths), desc='Segmenting images', unit='image'):
        mask = seg_per_frame(
                    segmentor, 
                    img_path,
                    batch_size
                )
        mask = mask[0]  # get the first mask, assuming single image input
        dst_img_path = save_raw_dir / f'{prefix}{idx:04d}.tif'
        imsave(str(dst_img_path), mask.astype(np.uint16))
        
        # correct mask
        dst_img_path = save_curated_dir / f'{prefix}filter_{idx:04d}.tif'
        empty_img = np.ones(shape=(1, 1))
        mask = correct_mask_fast(empty_img, mask, dst_img_path)

        
# def segment_fn_parallel(
#         input: Union[Path, List[Path]],
#         save_dir: Path,
#         device: str,
#         prefix: str = 'frame_',
#         batch_size: Optional[int] = None,
# ):
#     """Segment an image using the Cellpose model in parallel."""
#     if isinstance(input, Path):
#         input =  sorted(input.glob('*.tif'))
#     img_paths = input
#     dst_paths = [save_dir / f'{prefix}{i:04d}.tif' for i in range(len(img_paths))]