from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import tifffile as tiff

import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
from trackastra.data import example_data_bacteria

def cell_track(imgs: Union[Path, np.ndarray], masks: Union[Path, np.ndarray], device: str,
               out_dir: Path | None = None, 
               model_name: str = "general_2d", mode: str = "greedy"
            ):
    """Track cells in a sequence of images using the Trackastra model."""
    if isinstance(imgs, Path):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load some test data images and masks
    imgs, masks = example_data_bacteria()

    # Load a pretrained model
    model = Trackastra.from_pretrained("general_2d", device=device)

    # or from a local folder
    # model = Trackastra.from_folder('path/my_model_folder/', device=device)

    # Track the cells
    track_graph = model.track(imgs, masks, mode="greedy")  # or mode="ilp", or "greedy_nodiv"


    # Write to cell tracking challenge format
    ctc_tracks, masks_tracked = graph_to_ctc(
        track_graph,
        masks,
        outdir="tracked",
    )

    pass