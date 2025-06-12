from typing import Any, Optional, Union, List
from pathlib import Path
from tqdm import tqdm

import numpy as np

from cellpose import models
from cellpose.io import imread, imsave

def prepare(gpu=True):
    model = models.CellposeModel(gpu=gpu)
    return model

def seg_per_frame(
        segmentor, 
        img: Union[Path, np.ndarray],
) -> np.ndarray:
    if isinstance(img, Path):
        img = imread(img)
    assert isinstance(img, np.ndarray), "Input image must be a numpy array or a path to an image file."

    masks_pred, flows, styles = segmentor.eval(img, niter=1000)
    return masks_pred

def segment_fn(
        input: Union[Path, List[Path]],
        save_dir: Path,
        device: str,
        prefix: str = 'frame_',
):
    """Segment an image using the Cellpose model."""
    if isinstance(input, Path):
        input =  sorted(input.glob('*.tif'))
    img_paths = input

    segmentor = prepare(device)

    for idx, img_path in tqdm(enumerate(img_paths), desc='Segmenting images', unit='image'):
        mask = seg_per_frame(segmentor, img_path)
        dst_img_path = save_dir / f'{prefix}{idx:04d}.tif'
        imsave(str(dst_img_path), mask.astype(np.uint16))