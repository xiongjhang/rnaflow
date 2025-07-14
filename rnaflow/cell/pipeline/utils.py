import os
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Union, List, Tuple, Optional, Any
import logging
import sys
import importlib

from tqdm.auto import tqdm
import tifffile as tiff

def get_class(class_name: str, modules: List[str] = None) -> Any:
    '''Get a class by its name from a list of modules.'''
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

# region logging

loggers = {}

def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger

# region stack functions

def print_stack_info(stack: Union[np.ndarray, Path, str]):
    if isinstance(stack, (Path, str)):
        stack = tiff.imread(stack)

    print(f'strack shape: {stack.shape}')
    print(f'strack dtype: {stack.dtype}')

def stack_to_frames(input_data, output_dir, prefix='t'):
    '''Convert a 3D NumPy array or a TIFF file into individual TIFF frames and save them to a specified directory.

    Args:
        input_data (Union[str, Path, np.ndarray]): Input data can be a file path to a TIFF file or a 3D NumPy array.
        output_dir (str): Directory where the individual TIFF frames will be saved, always endwith '01/'.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(input_data, (str, Path)):
        print(f"Reading TIFF file from: {input_data}")
        tif_data = tiff.imread(input_data)
    elif isinstance(input_data, np.ndarray):
        print("Using provided NumPy array as input.")
        tif_data = input_data
    else:
        raise ValueError("input_data must be a file path (str or Path) or a 3D NumPy array.")
    
    if tif_data.ndim != 3:
        raise ValueError(f"input_data must be a 3D array, got shape {tif_data.shape}")

    for i, frame in tqdm(enumerate(tif_data), desc="Saving frames", unit="frame"):
        frame_filename = f"{prefix}{i:04d}.tif"
        frame_path = os.path.join(output_dir, frame_filename)
        
        tiff.imwrite(frame_path, frame)
        # print(f"Saved frame {i} to {frame_path}")

def frames_to_stack(folder_path):
    '''Concatenate all TIFF files in a specified folder into a single 3D NumPy array.'''
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    tif_files.sort()
    print(f'total files: {len(tif_files)}')
    
    images = []
    for tif_file in tif_files:
        file_path = os.path.join(folder_path, tif_file)
        img = tiff.imread(file_path)
        # print(f'file: {file_path}\nstack shape: {img.shape}')
        images.append(img)

    stack = np.stack(images, axis=0) if len(images) > 1 else images[0]
    return stack

def split_array_along_time(array, num_chunks):
    '''Split a 3D NumPy array into multiple chunks along the time dimension.'''
    num_frames = array.shape[0]
    chunk_size = num_frames // num_chunks
    remainder = num_frames % num_chunks

    chunk_indices = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_indices.append((start, end))
        start = end

    print(f'total frames: {num_frames}, chunk indices: {chunk_indices}')

    chunks = [array[start:end] for start, end in chunk_indices]
    return chunks


# region predictor 

class Predictor(ABC):

    @abstractmethod
    def prepare(self):
        """
        Prepare the predictor for use.
        This method should be called before any preprocessing or segmentation.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def preprocess(self):
        """
        Preprocess the input data using the provided preprocessing function.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def segment(self):
        """
        Segment the preprocessed data using the provided segmentation function.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def track(self):
        """
        Track the segmentation results.
        This method should be called after segmentation to track the results.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def postprocess(self):
        """
        Postprocess the segmentation results.
        This method should be called after tracking to finalize the results.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


def map_fn_to_frames(
        imgs_path_list: List[Path], 
        fn, 
        save_dir: Path, 
        save_prefix: str = 't', 
        **kwargs
) -> List:
    """
    Apply a function to each frame in a list of image paths.
    
    Args:
        imgs_path_list (List[Path]): List of paths to the image frames.
        fn (callable): Function to apply to each frame.
    
    Returns:
        List: List of results after applying the function to each frame.
    """
    # TODO: support parallel processing

    assert save_dir.is_dir(), f"save_dir must be a directory, got {save_dir}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(imgs_path_list)
    results = []
    for i, img_path in tqdm(enumerate(imgs_path_list), total=num_frames, desc='Processing frames', unit="frame"):
        img = tiff.imread(img_path)
        result = fn(img, **kwargs)
            
        save_path = save_dir / f'{save_prefix}_{i:04d}.tif'
        tiff.imwrite(save_path, result)
        results.append(result)
    return results  