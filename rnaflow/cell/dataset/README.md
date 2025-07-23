# Cell Dataset

Current public dataset for cell segmentation and tracking in fluorescence microscopy images.

- [Cell Tracking Challenge](https://celltrackingchallenge.net/)

- [DeepCell](https://deepcell.org/)


## Cell Tracking Dataset Labeling

Our original cell time-lapse dataset are sourced from [Single Molecule Dynamic Transcriptome Systems Biology Laboratory](https://www.westlake.edu.cn/faculty/yihan-wan.html). We label the cell tracking dataset using [micro-sam](https://github.com/computational-cell-analytics/micro-sam) based on [napari](https://github.com/napari/napari). For more information about the labeling tools, please refer to the [micro-sam documentation](https://computational-cell-analytics.github.io/micro-sam/).

The progress of our cell tracking dataset labeling are described as follows:

1. **Auto Segmentation:** we use [cellpose-sam](https://github.com/MouseLand/cellpose) to perform auto segmentation on the cell images with `vit-l-lm` model.

2. **Manual Interactive Correction:** we use [micro-sam annotation tools](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#annotation-tools) to perform manual interactive correction on the auto segmentation results. This step is to

    - remove false positive cell mask

    - add false negative cell mask

    - correct the cell mask shape

3. **Cell Tracking:** we use [trackastra](trackastra) to perform cell tracking on the corrected cell masks. This step is to 

    - using `greedy_nodiv` track mode to link the consecutive masks of individual cells across time frames, i.e., all cell's parent is `0`.

    - generate the cell tracking results in [cell tracking challenge format](https://celltrackingchallenge.net/datasets/).


> *Note*: In step 3, we foucs on the continuity and consistency of the cell masks across time frames, rather than the division event of the cells. This is because the cell division event is differ from which in public cell tracking datasets, e.g., ctc dataset. In our dataset, when the division event begins, the parent cell will be disappeared for a while (maybe dozens of frames), and then the daughter cells will appear near the parent cell. This is different from most of cell tracking dataset, where the parent cell will be divided into two daughter cells in the next frame. Therefore, we use `greedy_nodiv` track mode to link the consecutive masks of individual cells across time frames, because trackastra is not designed to handle such division events with a empty and long time gap.

4. **Cell Lineage Annotation:** we manually modify the daughter cell's parent id from `0` to the actual parent cell's id in the cell tracking results. This is to ensure that the daughter cells are correctly linked to their parent cells in the cell lineage annotation.
