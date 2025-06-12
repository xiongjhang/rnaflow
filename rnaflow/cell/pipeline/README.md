# Cell Segmentation and Tracking

- For segmentation, each method should realize following method:

    - `prepare(model_path, device, **kwargs)`: init model

    - `seg_per_frame(img, **kwargs)`: segment single frame

        - Return segmentation mask

        - Maybe input include information from previous frame, such as `prev_mask`, `prev_box`, etc.

    - `mask_filter(mask, **kwargs)`: filter segmentation mask

        - Return filtered mask

- For tracking, each method should realize following method:

    - `cell_track(img_dir, seg_dir, save_dir, **kwargs)`