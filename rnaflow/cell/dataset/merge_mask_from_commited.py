from pathlib import Path
from tqdm import tqdm

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def merge_mask_frame(
        existing_mask: np.ndarray, 
        additional_mask: np.ndarray,
        min_region_size: int = 200
) -> np.ndarray:
    """Merge two masks by adding the regions from the additional mask to the existing mask.

    This is inspired by the `commit` operation in micro-sam annoatation tools, for more details see 
    https://github.com/computational-cell-analytics/micro-sam/blob/7a3e3b539821275e757c10e3852c2c3e701d039b/micro_sam/sam_annotator/_widgets.py#L737

    Args:
        existing_mask (np.ndarray): The existing mask to which the additional mask will be merged.
        additional_mask (np.ndarray): The additional mask containing new regions to be added.   
    Raises:
        ValueError: If any region in the additional mask is smaller than 200 pixels.

    Returns:
        np.ndarray: The merged mask containing regions from both the existing and additional masks.
    """
    assert existing_mask.shape == additional_mask.shape, "Masks must have the same shape"

    merged_mask = existing_mask.copy()
    # Find the unique IDs in the existing mask
    # and determine the maximum ID to avoid conflicts with new IDs.
    all_existing_ids = np.unique(existing_mask)
    all_existing_ids = all_existing_ids[all_existing_ids != 0]
    max_existing_id = np.max(all_existing_ids) if len(all_existing_ids) > 0 else 0

    for frame_idx in tqdm(range(existing_mask.shape[0])):
        existing_frame = existing_mask[frame_idx]
        additional_frame = additional_mask[frame_idx]
        
        existing_ids = np.unique(existing_frame)
        existing_ids = existing_ids[existing_ids != 0]

        # Filter small regions which always not a full cell mask
        labeled_additional = label(additional_frame, connectivity=1)
        properties = regionprops(labeled_additional)
        for prop in properties:
            if prop.area < min_region_size:
                # labeled_additional[labeled_additional == prop.label] = 0
                raise ValueError(f"Frame {frame_idx} has a region smaller than {min_region_size} pixels, which is not allowed.")
        additional_regions = np.unique(labeled_additional)
        additional_regions = additional_regions[additional_regions != 0]

        if len(additional_regions) == 0:
            continue

        next_id = max(max_existing_id, np.max(existing_ids)) + 1 if len(existing_ids) > 0 else max_existing_id + 1
        id_mapping = {}
        
        for region_id in additional_regions:
            id_mapping[region_id] = next_id
            next_id += 1

        remapped_additional = np.zeros_like(labeled_additional)
        for old_id, new_id in id_mapping.items():
            remapped_additional[labeled_additional == old_id] = new_id

        # Merge the remapped additional mask into the existing mask,
        # the existing mask will be prioritized, so if a pixel.
        merged_mask[frame_idx] = np.where(
            (existing_frame == 0) & (remapped_additional != 0),
            remapped_additional,
            existing_frame
        )
    
    return merged_mask

if __name__ == "__main__":

    # Example Usage
    root_dir = Path("path/to/your/data")
    existing_mask_stack = tiff.imread(root_dir / "segment_res.tif").astype(np.uint16)
    committed_mask_stack = tiff.imread(root_dir / "committed_objects.tif").astype(np.uint16)   

    merged_mask_stack = merge_mask_frame(existing_mask_stack, committed_mask_stack)
    tiff.imwrite(root_dir / "merged_mask_stack.tif", merged_mask_stack.astype(np.uint16), imagej=True)