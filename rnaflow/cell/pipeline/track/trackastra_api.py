from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import tifffile as tiff

import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks

def cell_track(
        imgs: Union[Path, np.ndarray],
        masks: Union[Path, np.ndarray],
        device: str,
        out_dir: Path | None = None, 
        model_name: str = "general_2d",
        mode: str = "greedy"
    ) -> tuple[pd.DataFrame, np.ndarray]:
    """Track cells in a sequence of images using the Trackastra model."""
    if isinstance(imgs, Path):
        assert imgs.is_dir() and masks.is_dir(), "imgs and masks should be directories containing TIFF files"
        paths = sorted(imgs.glob('*.tif'))
        imgs = np.stack([tiff.imread(p) for p in paths], axis=0)
        paths = sorted(masks.glob('*.tif'))
        masks = np.stack([tiff.imread(p) for p in paths], axis=0)

    assert imgs.ndim == 3 and imgs.shape[0] > 1, "Input images should be a 3D array (time, height, width)"
    assert imgs.shape == masks.shape, "Input images and masks should have the same shape"

    model = Trackastra.from_pretrained(model_name, device=device)
    track_graph = model.track(imgs, masks, mode=mode)  # or mode="ilp", or "greedy_nodiv"
    ctc_tracks, masks_tracked = graph_to_ctc(
        track_graph,
        masks,
        outdir=out_dir,
    )
    return ctc_tracks, masks_tracked

if __name__ == "__main__":
    # Example usage
    root_dir = Path("path/to/data")
    img_stack = tiff.imread(root_dir / "images.tif")
    mask_strack = tiff.imread(root_dir / "masks.tif")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    track_mode = "greedy_nodiv"  # or "greedy", "ilp"
    dst_dir = root_dir / f'trackastra_{track_mode}'
    dst_file = str(dst_dir) + '.tif'
    
    ctc_tracks, masks_tracked = cell_track(
        imgs=img_stack,
        masks=mask_strack,
        device=device,
        out_dir=dst_dir,
        model_name="general_2d",
        mode=track_mode
    )
    
    tiff.imwrite(dst_file, masks_tracked)