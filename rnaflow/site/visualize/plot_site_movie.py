from typing import Dict, Optional, Tuple, Literal
from os import makedirs
from os.path import join
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

np.random.seed(0)
PALETTE = np.random.randint(0, 256, (10000, 3))


def get_palette_color(i):
    i = i % PALETTE.shape[0]
    return PALETTE[i]

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame to 0-255 range"""
    frame = frame.astype(np.float32)
    frame = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-5)
    return (frame * 255).astype(np.uint8)

def create_colored_image(
        img: np.ndarray,
        trajectories: np.ndarray,
        site_id: int,
        frame: int = None,
        plot_frame: bool = False,
        alpha: float = 1,
        draw_mode: str = None,
        trajectory_thickness: int = 2,
        box_size: int = 8,
        box_thickness: int = 1,
):
    """
    Creates an image with trajectories drawn on top of the input image.

    Args:
        img: np.ndarray
            The input image.
        trajectories: np.ndarray
            3D array of shape (N, 3) where N is the length of the trajectory.
            Each row contains the (x, y, t) coordinates of a trajectory point.
        site_id: int
            The site ID for which the trajectory is being visualized.
        frame: int
            The frame number.
        plot_frame: bool
            Whether to plot the frame number on the image.
        alpha: float
            Transparency factor for the overlay.
        draw_mode: str
            Drawing mode: 'line' to draw trajectory lines, 'box' to draw boxes at each point, 
                        'line&box' to draw both lines and boxes.    
        trajectory_thickness: int
            Thickness of the trajectory lines.
        box_size: int
            Size of the boxes to draw around trajectory points.
        box_thickness: int
            Thickness of the box edges.

    Returns:
        The colored image.
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    color = get_palette_color(site_id).tolist()

    h, w = img.shape[:2]
    traj_layer = np.zeros((h, w, 3), dtype=np.uint8)

    if trajectories is not None and len(trajectories) > 0:
        # Draw trajectories
        if draw_mode in ['line', 'line&box'] and len(trajectories) > 1:
            for i in range(1, len(trajectories)):
                pt1 = tuple(np.round(trajectories[i - 1][:2]).astype(int))
                pt2 = tuple(np.round(trajectories[i][:2]).astype(int))
                thickness = max(1, int(trajectory_thickness * (i / len(trajectories))))
                cv2.line(traj_layer, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        if draw_mode in ['box', 'line&box']:
            latest_pt = tuple(np.round(trajectories[-1]).astype(int))
            if latest_pt[-1] == frame:
                top_left = (latest_pt[0] - box_size // 2, latest_pt[1] - box_size // 2)
                bottom_right = (latest_pt[0] + box_size // 2, latest_pt[1] + box_size // 2)
                cv2.rectangle(traj_layer, top_left, bottom_right, color, thickness=box_thickness)
    
    blended = cv2.addWeighted(img, 1.0, traj_layer, alpha, 0)

    # Add frame number if specified
    if plot_frame is not None:
        cv2.putText(img, str(frame), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return blended

def visualize(
        img: np.ndarray,
        img_reg: np.ndarray,
        traj_data: dict[int, pd.DataFrame],
        viz_dir: str = None,
        video_name: str = None,
        alpha: float = 1,
        draw_mode: Literal['line', 'box', 'line&box'] = 'line',
        trajectory_length: int = 10,
        trajectory_thickness: int = 2,
        framerate: int = 30,
        using_tp: bool = False,
        resize_factor: int = 1,
        add_reg_space: bool = True,
):
    """
    Visualize the cell trajectory on the cell sequence.

    Args:
        img : np.ndarray
            The original cell sequence.
        img_reg : np.ndarray
            The registered cell sequence.
        traj_data : dict[int, pd.DataFrame]
            Trajectory data for each site, where the key is the site ID and the value
            is a DataFrame containing trajectory information, required columns are:
            - 'Org_X, Org_Y': Coordinates of the trajectory points in the original space.
            - 'Reg_X, Reg_Y': Coordinates of the trajectory points in the registered space.
            - 'TP_Flag': A flag indicating whether the trajectory point is a visualization site.
        viz_dir : str, optional
            The directory to save the visualization. If None, no visualization is saved.
        alpha : float, default 1
            Transparency factor for the overlay.
        draw_mode : Literal['line', 'box',  'line&box'], default 'line'
            The drawing mode for the trajectory:
            - 'line': Draw lines connecting trajectory points.
            - 'box': Draw boxes around trajectory points.
            - 'line&box': Draw both lines and boxes.
        video_name : str, optional
            The name of the video file to save. If None, no video will be created.
        trajectory_length : int, default 10
            The length of the trajectory to visualize.
        trajectory_thickness : int, default 2
            The thickness of the trajectory lines.
        framerate : int, default 30
            The framerate of the video if created.
        using_tp : bool, default False
            Whether to use the 'TP_Flag' column in the trajectory data to filter points.
            Only affective if `draw_mode` is 'box'.
        resize_factor : int, default 1
            Factor by which to resize the images for visualization.
        add_reg_space : bool, default True
            Whether to add the registered space image next to the original image in the visualization.
    """
    # Check input validity
    assert img.shape == img_reg.shape, \
        f"Original and registered images must have the same shape, but got {img.shape} and {img_reg.shape}."
    assert img.ndim == 3, f"Input images must be 3-dimensional, but got {img.ndim} dimensions."
    assert len(traj_data) > 0, "Trajectory data must not be empty."

    # Combined example
    img_0 = cv2.resize(img[0], (img[0].shape[1] * resize_factor, img[0].shape[0] * resize_factor),
                       interpolation=cv2.INTER_LINEAR)
    img_reg_0 = cv2.resize(img_reg[0], (img_reg[0].shape[1] * resize_factor, img_reg[0].shape[0] * resize_factor),
                           interpolation=cv2.INTER_LINEAR)
    combined_frame = np.hstack((img_0, img_reg_0)) if add_reg_space else img_0

    # Create visualization directory
    if viz_dir:
        makedirs(viz_dir, exist_ok=True)

    video_writer = None

    # Initialize trajectory tracking
    trajectory_history = defaultdict(lambda: {'org': [], 'reg': []})
    max_trajectory_length = trajectory_length
    base_trajectory_thickness = trajectory_thickness

    for site_id, df in traj_data.items():
        print(f"\nProcessing site {site_id}...")
        
        # Initialize video writer for this site
        video_format = 'mp4'
        if video_name:
            site_video_path = join(viz_dir, f"{video_name}_{site_id}_{draw_mode}.{video_format}")
            fourcc = {
                'mp4': cv2.VideoWriter_fourcc(*'mp4v'),
                'avi': cv2.VideoWriter_fourcc(*'XVID')
            }.get(video_format, cv2.VideoWriter_fourcc(*'mp4v'))
            
            video_writer = cv2.VideoWriter(
                site_video_path, 
                fourcc, 
                framerate, 
                (combined_frame.shape[1], combined_frame.shape[0]))
        else:
            video_writer = None
        
        # Initialize trajectory history
        org_traj = []
        reg_traj = []

        # Process each frame
        for t in range(len(img)):
            # Visualize the image   
            frame_org = normalize_frame(img[t])
            frame_reg = normalize_frame(img_reg[t])
            frame_org = cv2.cvtColor(frame_org, cv2.COLOR_GRAY2BGR)
            frame_reg = cv2.cvtColor(frame_reg, cv2.COLOR_GRAY2BGR)
            frame_org = cv2.resize(frame_org, (frame_org.shape[1] * resize_factor, frame_org.shape[0] * resize_factor),
                                  interpolation=cv2.INTER_LINEAR)
            frame_reg = cv2.resize(frame_reg, (frame_reg.shape[1] * resize_factor, frame_reg.shape[0] * resize_factor),
                                    interpolation=cv2.INTER_LINEAR)
                
            # Get trajectory up to current frame
            if using_tp:
                df_frame = df[(df.index <= t) & (df['TP_Flag'] == 1)].tail(max_trajectory_length)
            else:
                df_frame = df[df.index <= t].tail(max_trajectory_length)
            
            if len(df_frame) > 0:
                org_traj = (df_frame[['Org_X', 'Org_Y','POSITION_T']].values * resize_factor).astype(int)
                reg_traj = (df_frame[['Reg_X', 'Reg_Y','POSITION_T']].values * resize_factor).astype(int)

            viz_org = create_colored_image(
                frame_org, org_traj, site_id,
                frame=t, 
                plot_frame=False,
                alpha=alpha, draw_mode=draw_mode,
                trajectory_thickness=base_trajectory_thickness
            )
            
            if add_reg_space:
                viz_reg = create_colored_image(
                    frame_reg, reg_traj, site_id,
                    frame=t, 
                    plot_frame=False,
                    alpha=alpha, draw_mode=draw_mode,
                    trajectory_thickness=base_trajectory_thickness
                )

            # Combine the visualized frames
            combined_frame = np.hstack((viz_org, viz_reg)) if add_reg_space else viz_org

            # Save the visualized frames
            if video_writer:
                video_writer.write(combined_frame)


# ====== Other functions for visualization ======


def main():

    root = '/mnt/sda/xjh/dataset/cell-data/20250725-xiangyu_vis/cellraw_4'
    cell_list = ['0', 'A', 'B', 'C', 'D', 'F']

    for char in cell_list:
        data_dir = join(root, f'cellraw_4-{char}')


        # data_dir = '/mnt/sda/xjh/dataset/site-data/20250721-xiangyu_vis/cellraw_12250'
        img = tiff.imread(join(data_dir, 'imgs_raw_mask.tif'))[0]
        img_reg = tiff.imread(join(data_dir, 'imgs_raw_mask_reg_rcs.tif'))[0]
        traj_data = {
            0: pd.read_csv(join(data_dir, 'dataAnalysis_tj_0_withBg.csv')),
        }

        visualize(
            img,
            img_reg,
            traj_data,
            viz_dir=data_dir,
            video_name='site_trajectory_visualization',
            draw_mode='line&box',
            trajectory_length=1000,
            trajectory_thickness=2,
            framerate=10,
            using_tp=True,
            add_reg_space=True
        )

if __name__ == "__main__":
    main()