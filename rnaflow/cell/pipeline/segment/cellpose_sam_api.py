from typing import Any, Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import logging

import numpy as np

from cellpose import models
from cellpose.io import imread, imsave

from rnaflow.cell.pipeline.utils import get_logger

logger = get_logger(__name__, level=logging.INFO)

def prepare(gpu=True):
    logger.info(f"Loading Cellpose model...")
    model = models.CellposeModel(gpu=gpu)
    return model

def seg_per_frame(
        segmentor, 
        img: Union[Path, List[Path], np.ndarray],
        batch_infer: bool = False,
) -> np.ndarray:

    if isinstance(img, Path) and not batch_infer:
        input = imread(img)
    elif isinstance(img, list) and not batch_infer:
        input = [imread(p) for p in img]
    elif isinstance(img, list) and batch_infer:
        imgs = [imread(p) for p in img]
        input = np.stack(imgs, axis=0)
    elif isinstance(img, np.ndarray):
        input = img
    else:
        raise ValueError(f"Input type {type(img)} is conflicting with batch_infer={batch_infer}. ")

    # input can be list of 2D/3D/4D images, or array of 2D/3D/4D images.
    masks_pred, flows, styles = segmentor.eval(input, niter=1000)
    if isinstance(masks_pred, list):
        masks_pred = np.stack(masks_pred, axis=0)

    return masks_pred

def segment_fn(
        input: Union[Path, List[Path]],
        save_dir: Path,
        device: str,
        prefix: str = 'frame_',
        batch_size: Optional[int] = None,
):
    """Segment an image using the Cellpose model."""
    if isinstance(input, Path):
        input =  sorted(input.glob('*.tif'))
    img_paths = input
    segmentor = prepare()

    # TODO: support mask filter
    # TODO: support batch inference
    for idx, img_path in tqdm(enumerate(img_paths), desc='Segmenting images', unit='image'):
        mask = seg_per_frame(segmentor, img_path)
        dst_img_path = save_dir / f'{prefix}{idx:04d}.tif'
        imsave(str(dst_img_path), mask.astype(np.uint16))