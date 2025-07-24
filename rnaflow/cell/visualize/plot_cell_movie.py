'''https://github.com/CellTrackingChallenge/py-ctcmetrics/blob/main/ctc_metrics/scripts/visualize.py

When process very big images, it will be very slow, e.g., for a 6997*2048 img,
it will take avout 8s to update the trajectories and 15s to create the visualization.
- in `create_colored_image`, most of time is spent on drawing objects than trajectories
- when `labels` and `parents` are set to False

This two step is the bottleneck of the visualization.

*Note:* Progress time is also affected by the number of objects in the image and 
the length of the trajectories of each object.

After optimization, 
 - it will take about 0.1s to update the trajectories
 - it will take about 9s to create the visualization
'''

import argparse
from typing import Literal, Optional
from os import listdir, makedirs
from os.path import join
import tifffile as tiff
import cv2
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import timeit
import time
from skimage import measure

from ctc_metrics.utils.filesystem import read_tracking_file


SHOW_BORDER = True
BORDER_WIDTH = {
    "BF-C2DL-HSC": 25,
    "BF-C2DL-MuSC": 25,
    "Fluo-N2DL-HeLa": 25,
    "PhC-C2DL-PSC": 25,
    "Fluo-N2DH-SIM+": 0,
    "DIC-C2DH-HeLa": 50,
    "Fluo-C2DL-Huh7": 50,
    "Fluo-C2DL-MSC": 50,
    "Fluo-N2DH-GOWT1": 50,
    "PhC-C2DH-U373": 50,
}

np.random.seed(0)
PALETTE = np.random.randint(0, 256, (10000, 3))


def get_palette_color(i):
    i = i % PALETTE.shape[0]
    return PALETTE[i]

def load_frame_data(img_path, res_path):
    img = tiff.imread(img_path).squeeze()
    assert img.ndim == 2, "Image must be 2D (single channel)"
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip((img - p1) / max(p99 - p1, 1e-5) * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    res_img = tiff.imread(res_path).squeeze()
    assert res_img.ndim == 2, "Result image must be 2D (single channel)"
    return img, res_img

def create_colored_image(
        img: np.ndarray,
        res: np.ndarray,
        labels: bool = False,
        opacity: float = 0.5,
        ids_to_show: list = None,
        frame: int = None,
        parents: dict = None,
        trajectories: dict = None,
        trajectory_thickness: int = 2
):
    """
    Creates a colored image from the input image and the results.

    Args:
        img: np.ndarray
            The input image.
        res: np.ndarray
            The results.
        labels: bool
            Print instance labels to the output.
        opacity: float
            The opacity of the instance colors.
        ids_to_show: list
            The IDs of the instances to show. All others will be ignored.
        frame: int
            The frame number.
        parents: dict
            The parent dictionary.
        trajectories: dict
            Dictionary of trajectories for each object ID.
        trajectory_thickness: int
            Thickness of the trajectory lines.
    Returns:
        The colored image.
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)

    uniqu_ids = np.unique(res)
    valid_ids = uniqu_ids[uniqu_ids > 0]
    if ids_to_show is not None:
        valid_ids = np.intersect1d(valid_ids, ids_to_show)
    palette_colors = {i: get_palette_color(i) for i in valid_ids}
    
    # Draw trajectories first (so they appear behind the objects)
    start = time.time()
    if trajectories is not None:
        for obj_id, points in trajectories.items():
            if obj_id not in valid_ids:
                continue
            if len(points) > 1:
                color = palette_colors[obj_id].tolist()
                for i in range(1, len(points)):
                    thickness = max(1, int(trajectory_thickness * (i / len(points))))
                    cv2.line(img, points[i-1], points[i], color, thickness)
    print(f"Plot- draw trajectories drawn in {time.time() - start:.2f} seconds.")
    
    # Draw objects
    start = time.time()
    for i in valid_ids:
        mask = res == i
        contour = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
        color = palette_colors[i]
        img[mask] = (
            np.round((1 - opacity) * img[mask] + opacity * color)
        )
        img[contour] = color
        if frame is not None:
            cv2.putText(img, str(frame), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if labels:
            # Print label to the center of the object
            y, x = np.where(mask)
            y, x = np.mean(y), np.mean(x)
            text = str(i)
            if parents is not None:
                if i in parents:
                    if parents[i] != 0:
                        text += f"({parents[i]})"
            cv2.putText(img, text, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    print(f"Plot- draw objects drawn in {time.time() - start:.2f} seconds.")
    return img

def visualize(
        img_dir: str,
        res_dir: str,
        viz_dir: str = None,
        video_name: str = None,
        video_format: Literal['mp4', 'avi'] = 'mp4',
        border_width: str = None,
        show_labels: bool = True,
        show_parents: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 10,
        trajectory_thickness: int = 2,
        ids_to_show: list = None,
        start_frame: int = 0,
        end_frame: int = None,
        framerate: int = 30,
        opacity: float = 0.5,
):  # pylint: disable=too-many-arguments,too-complex,too-many-locals
    """
    Visualizes the tracking results.

    Args:
        img_dir: str
            The path to the images.
        res_dir: str
            The path to the results.
        viz_dir: str
            The path to save the visualizations.
        video_name: str
            The base name of the video file to save. If None, no video will be created.
            Note that no visualization is available during video creation.
        video_format: Literal['mp4', 'avi']
            The format of the video to save. Default is 'mp4'.
        border_width: str or int
            The width of the border. Either an integer or a string that
            describes the challenge name.
        show_labels: bool
            Print instance labels to the output.
        show_parents: bool
            Print parent labels to the output.
        show_trajectories: bool
            Show movement trajectories of the instances.
        trajectory_length: int
            Number of previous frames to show in the trajectory.
        trajectory_thickness: int
            Thickness of the trajectory lines.
        ids_to_show: list
            The IDs of the instances to show. All others will be ignored.
        start_frame: int
            The frame to start the visualization.
        end_frame: int
            The frame to end the visualization. If None, visualizes all frames.
        framerate: int
            The framerate of the video.
        opacity: float
            The opacity of the instance colors.

    """
    # Define initial video parameters
    wait_time = max(1, round(1000 / framerate))
    if border_width is None:
        border_width = 0
    elif isinstance(border_width, str):
        try:
            border_width = int(border_width)
        except ValueError as exc:
            if border_width in BORDER_WIDTH:
                border_width = BORDER_WIDTH[border_width]
            else:
                raise ValueError(
                    f"Border width '{border_width}' not recognized. "
                    f"Existing datasets: {BORDER_WIDTH.keys()}"
                ) from exc

    # Load image and tracking data
    start = time.time()
    images = [x for x in sorted(listdir(img_dir)) if x.endswith(".tif")][:20]
    results = [x for x in sorted(listdir(res_dir)) if x.endswith(".tif")][:20]
    tracking_data = read_tracking_file(join(res_dir, "res_track.txt"))
    parents = {l[0]: l[3] for l in tracking_data}
    print(f"Loaded {len(images)} images and {len(results)} results in {time.time() - start:.2f} seconds.")

    # Create visualization directory
    if viz_dir:
        makedirs(viz_dir, exist_ok=True)

    video_writer = None
    
    # Initialize trajectory tracking
    trajectory_history = defaultdict(list)
    max_trajectory_length = trajectory_length
    base_trajectory_thickness = trajectory_thickness

    # Loop through all images
    while start_frame < len(images):
        # Read image file
        img_name, res_name = images[start_frame], results[start_frame]
        img_path, res_path,  = join(img_dir, img_name), join(res_dir, res_name)
        print(f"\rFrame {img_name} (of {len(images)})", end="")

        # Visualize the image
        start = time.time()
        img, res_img = load_frame_data(img_path, res_path)
        print(f"Loaded in {time.time() - start:.2f} seconds.")
        
        # Update trajectory history
        start = time.time()
        if show_trajectories:
            current_centers = {}
            props = measure.regionprops(res_img.astype(int))
            for prop in props:
                obj_id = prop.label
                if obj_id == 0:
                    continue
                center = (int(prop.centroid[1]), int(prop.centroid[0]))  # (x,y)
                current_centers[obj_id] = center

            existing_ids = set(trajectory_history.keys())
            current_ids = set(current_centers.keys())

            # Remove trajectories for objects that disappeared
            for obj_id in list(existing_ids - current_ids):
                del trajectory_history[obj_id]

            # Update existing trajectories
            for obj_id in list(existing_ids & current_ids):
                trajectory_history[obj_id].append(current_centers[obj_id])
                # Trim to max length
                if len(trajectory_history[obj_id]) > max_trajectory_length:
                    trajectory_history[obj_id] = trajectory_history[obj_id][-max_trajectory_length:]

            # Add new objects
            for new_id in current_ids - existing_ids:
                trajectory_history[new_id] = [current_centers[new_id]]
        print(f"Updated trajectories in {time.time() - start:.2f} seconds.")

        start = time.time()
        viz = create_colored_image(
            img,
            res_img,
            labels=show_labels,
            frame=start_frame,
            parents=parents if show_parents else None,
            ids_to_show=ids_to_show,
            opacity=opacity,
            trajectories=trajectory_history if show_trajectories else None,
            trajectory_thickness=trajectory_thickness
        )
        print(f"Created visualization in {time.time() - start:.2f} seconds.")
        
        if border_width > 0:
            viz = cv2.rectangle(
                viz,
                (border_width, border_width),
                (viz.shape[1] - border_width, viz.shape[0] - border_width),
                (0, 0, 255), 1
            )

        # Save the visualization
        start = time.time()
        if video_name is not None:
            if video_writer is None:
                video_basename = video_name.split('.')[0]  # remove available extension
                video_path = join(viz_dir, f"{video_basename}.{video_format}")

                # choose fourcc encoding based on video format
                if video_format == 'mp4':
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # H.264 encoding
                elif video_format == 'avi':
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI encoding
                else:
                    raise ValueError(f"Unsupported video format: {video_format}")

                video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    framerate,
                    (viz.shape[1], viz.shape[0])
                )
            video_writer.write(viz)
            print(f"Saved visualization in {time.time() - start:.2f} seconds.")
            start_frame += 1
            continue

        # Show the video
        cv2.imshow("VIZ", viz)
        key = cv2.waitKey(wait_time)
        if key == ord("q"):
            # Quit the visualization
            break
        if key == ord("w"):
            # Start or stop the auto visualization
            if wait_time == 0:
                wait_time = max(1, round(1000 / framerate))
            else:
                wait_time = 0
        elif key == ord("d"):
            # Move to the next frame
            start_frame += 1
            wait_time = 0
        elif key == ord("a"):
            # Move to the previous frame
            start_frame -= 1
            wait_time = 0
        elif key == ord("l"):
            # Toggle the show labels option
            show_labels = not show_labels
        elif key == ord("p"):
            # Toggle the show parents option
            show_parents = not show_parents
        elif key == ord("t"):
            # Toggle the show trajectories option
            show_trajectories = not show_trajectories
        elif key == ord("+"):
            # Increase trajectory length
            max_trajectory_length += 5
        elif key == ord("-"):
            # Decrease trajectory length
            max_trajectory_length = max(0, max_trajectory_length - 5)
        elif key == ord("]"):  
            base_trajectory_thickness = min(10, base_trajectory_thickness + 1)
        elif key == ord("["): 
            base_trajectory_thickness = max(1, base_trajectory_thickness - 1)
        elif key == ord("s"):
            # Save the visualization
            if viz_dir is None:
                print("Please define the '--viz' argument to save the "
                      "visualizations.")
                continue
            viz_path = join(viz_dir, img_name) + ".jpg"
            cv2.imwrite(viz_path, viz)
        else:
            # Move to the next frame
            start_frame += 1

def visualize_parallel(
        img_dir: str,
        res_dir: str,
        viz_dir: str = None,
        video_name: str = None,
        video_format: Literal['mp4', 'avi'] = 'mp4',
        border_width: str = None,
        show_labels: bool = True,
        show_parents: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 10,
        trajectory_thickness: int = 2,
        ids_to_show: list = None,
        start_frame: int = 0,
        framerate: int = 30,
        opacity: float = 0.5,
):  # pylint: disable=too-many-arguments,too-complex,too-many-locals
    """
    Visualizes the tracking results.

    Args:
        img_dir: str
            The path to the images.
        res_dir: str
            The path to the results.
        viz_dir: str
            The path to save the visualizations.
        video_name: str
            The base name of the video file to save. If None, no video will be created.
            Note that no visualization is available during video creation.
        video_format: Literal['mp4', 'avi']
            The format of the video to save. Default is 'mp4'.
        border_width: str or int
            The width of the border. Either an integer or a string that
            describes the challenge name.
        show_labels: bool
            Print instance labels to the output.
        show_parents: bool
            Print parent labels to the output.
        show_trajectories: bool
            Show movement trajectories of the instances.
        trajectory_length: int
            Number of previous frames to show in the trajectory.
        trajectory_thickness: int
            Thickness of the trajectory lines.
        ids_to_show: list
            The IDs of the instances to show. All others will be ignored.
        start_frame: int
            The frame to start the visualization.
        framerate: int
            The framerate of the video.
        opacity: float
            The opacity of the instance colors.

    """
    # Define initial video parameters
    wait_time = max(1, round(1000 / framerate))
    if border_width is None:
        border_width = 0
    elif isinstance(border_width, str):
        try:
            border_width = int(border_width)
        except ValueError as exc:
            if border_width in BORDER_WIDTH:
                border_width = BORDER_WIDTH[border_width]
            else:
                raise ValueError(
                    f"Border width '{border_width}' not recognized. "
                    f"Existing datasets: {BORDER_WIDTH.keys()}"
                ) from exc

    # Load image and tracking data
    images = [x for x in sorted(listdir(img_dir)) if x.endswith(".tif")]
    results = [x for x in sorted(listdir(res_dir)) if x.endswith(".tif")]
    tracking_data = read_tracking_file(join(res_dir, "res_track.txt"))
    parents = {l[0]: l[3] for l in tracking_data}

    # Create visualization directory
    if viz_dir:
        makedirs(viz_dir, exist_ok=True)

    video_writer = None
    
    # Initialize trajectory tracking
    trajectory_history = defaultdict(list)
    max_trajectory_length = trajectory_length
    base_trajectory_thickness = trajectory_thickness

    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Optional[Future] = None

        # Loop through all images
        while start_frame < len(images):
            # Read image file
            img_name, res_name = images[start_frame], results[start_frame]
            img_path, res_path,  = join(img_dir, img_name), join(res_dir, res_name)
            print(f"\rFrame {img_name} (of {len(images)})", end="")

            # Load the next frame asynchronously
            if future is None:
                # Load the next frame data
                future = executor.submit(load_frame_data, img_path, res_path)
            
            img, res_img = future.result()

            # Submit the next frame for loading
            next_frame = start_frame + 1
            if next_frame < len(images):
                next_img_name, next_res_name = images[next_frame], results[next_frame]
                next_img_path, next_res_path = join(img_dir, next_img_name), join(res_dir, next_res_name)
                future = executor.submit(load_frame_data, next_img_path, next_res_path)
            else:
                future = None
            
            # Update trajectory history
            if show_trajectories:
                current_centers = {}
                for i in np.unique(res_img):
                    if i == 0:
                        continue
                    mask = res_img == i
                    y, x = np.where(mask)
                    center = (int(np.mean(x)), int(np.mean(y)))
                    current_centers[i] = center
                    
                # Update history for existing trajectories
                for obj_id in list(trajectory_history.keys()):
                    if obj_id in current_centers:
                        trajectory_history[obj_id].append(current_centers[obj_id])
                        # Trim to max length
                        if len(trajectory_history[obj_id]) > max_trajectory_length:
                            trajectory_history[obj_id] = trajectory_history[obj_id][-max_trajectory_length:]
                    else:
                        # Remove trajectories for objects that disappeared
                        del trajectory_history[obj_id]
                
                # Add new objects
                for obj_id in current_centers:
                    if obj_id not in trajectory_history:
                        trajectory_history[obj_id] = [current_centers[obj_id]]

            viz = create_colored_image(
                img,
                res_img,
                labels=show_labels,
                frame=start_frame,
                parents=parents if show_parents else None,
                ids_to_show=ids_to_show,
                opacity=opacity,
                trajectories=trajectory_history if show_trajectories else None,
                trajectory_thickness=trajectory_thickness
            )
            
            if border_width > 0:
                viz = cv2.rectangle(
                    viz,
                    (border_width, border_width),
                    (viz.shape[1] - border_width, viz.shape[0] - border_width),
                    (0, 0, 255), 1
                )

            # Save the visualization
            if video_name is not None:
                if video_writer is None:
                    video_basename = video_name.split('.')[0]  # remove available extension
                    video_path = join(viz_dir, f"{video_basename}.{video_format}")

                    # choose fourcc encoding based on video format
                    if video_format == 'mp4':
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # H.264 encoding
                    elif video_format == 'avi':
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI encoding
                    else:
                        raise ValueError(f"Unsupported video format: {video_format}")

                    video_writer = cv2.VideoWriter(
                        video_path,
                        fourcc,
                        framerate,
                        (viz.shape[1], viz.shape[0])
                    )
                video_writer.write(viz)
                start_frame += 1
                continue


def parse_args():
    """ Parses the arguments. """
    parser = argparse.ArgumentParser(description='Validates CTC-Sequences.')
    parser.add_argument(
        '--img', type=str, required=True,
        help='The path to the images.'
    )
    parser.add_argument(
        '--res', type=str, required=True, help='The path to the results.'
    )
    parser.add_argument(
        '--viz', type=str, default=None,
        help='The path to save the visualizations.'
    )
    parser.add_argument(
        '--video-name', type=str, default=None,
        help='The path to the video if a video should be created. Note that no '
             'visualization is available during video creation.'
    )
    parser.add_argument(
        '--video-format', type=str, default="mp4", choices=["mp4", "avi"],
        help='Video format to save (mp4 or avi). Default: mp4.'
    )
    parser.add_argument(
        '--border-width', type=str, default=None,
        help='The width of the border. Either an integer or a string that '
             'describes the challenge name.'
    )
    parser.add_argument(
        '--show-no-labels', action="store_false",
        help='Print no instance labels to the output.'
    )
    parser.add_argument(
        '--show-no-parents', action="store_false",
        help='Print no parent labels to the output.'
    )
    parser.add_argument(
        '--show-no-trajectories', action="store_false",
        help='Do not show movement trajectories.'
    )
    parser.add_argument(
        '--trajectory-length', type=int, default=10,
        help='Number of previous frames to show in the trajectory.'
    )
    parser.add_argument(
        '--trajectory-thickness', type=int, default=2,
        help='Thickness of the trajectory lines.'
    )
    parser.add_argument(
        '--ids-to-show', type=int, nargs='+', default=None,
        help='The IDs of the instances to show. All others will be ignored.'
    )
    parser.add_argument(
        '--start-frame', type=int, default=0,
        help='The frame to start the visualization.'
    )
    parser.add_argument(
        '--framerate', type=int, default=10,
        help='The framerate of the video.'
    )
    parser.add_argument(
        '--opacity', type=float, default=0.5,
        help='The opacity of the instance colors.'
    )
    args = parser.parse_args()
    return args


def main():
    """
    Main function that is called when the script is executed.
    """
    # args = parse_args()

    # Test dataset
    img = r'D:\dataset\cell-benchmark\Fluo-N2DH-SIM+\01'
    res = r'D:\dataset\cell-benchmark\Fluo-N2DH-SIM+\01_GT\TRA'
    viz_dir = r'D:\dataset\cell-benchmark\Fluo-N2DH-SIM+\01_VIS'

    # Yokogawa dataset
    img = r'D:\dataset\cell-data\vis_xiangyu\240129D0_F007\01'
    res = r'D:\dataset\cell-data\vis_xiangyu\240129D0_F007\01_GT\_RES'
    viz_dir = r'D:\dataset\cell-data\vis_xiangyu\240129D0_F007\01_VIS'

    # LLS dataset
    img = '/mnt/sda/cell_data/LLS_SOX2_20240904/LLS_SOX2_01/01'
    res = '/mnt/sda/cell_data/LLS_SOX2_20240904/LLS_SOX2_01/01_GT/_RES'
    viz_dir = '/mnt/sda/xjh/dataset/cell-data/20250722-xiangyu_vis/01_VIS'

    kwargs = dict(
        img_dir=img,
        res_dir=res,
        viz_dir=viz_dir,
        # video_name='01_video_20f',
        video_format='mp4',
        border_width=None,
        show_labels=False,
        show_parents=False,
        show_trajectories=True,
        trajectory_length=150,
        trajectory_thickness=8,
        ids_to_show=None,
        start_frame=0,
        framerate=10,
        opacity=0.5,
    )
    video_name = '01_video_20f_test'
    using_parallel = False  # Set to True to use parallel visualization

    start_time = timeit.default_timer()

    if not using_parallel:
        visualize(
            video_name=video_name, 
            **kwargs
        )
    else:
        visualize_parallel(
            video_name=video_name+'_parallel',
            **kwargs
        )
    
    end_time = timeit.default_timer()
    print(f'Parallel visualization: {using_parallel}')
    print(f"\nVisualization completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()