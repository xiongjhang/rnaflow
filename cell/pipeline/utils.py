import os
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Union

from tqdm.auto import tqdm
import tifffile as tiff

# region stack functions

def print_stack_info(stack: Union[np.ndarray, Path, str]):
    if isinstance(stack, (Path, str)):
        stack = tiff.imread(stack)

    print(f'strack shape: {stack.shape}')
    print(f'strack dtype: {stack.dtype}')

def stack_to_frames(input_data, output_dir):
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
        raise ValueError("输入必须是文件路径（字符串）或三维 NumPy 数组。")
    
    if tif_data.ndim != 3:
        raise ValueError(f"输入数据必须是三维数组，形状为 (num_frames, height, width)。{tif_data.shape}")

    for i, frame in tqdm(enumerate(tif_data), desc="Saving frames", unit="frame"):
        frame_filename = f"t{i:04d}.tif"
        frame_path = os.path.join(output_dir, frame_filename)
        
        tiff.imwrite(frame_path, frame)
        # print(f"Saved frame {i} to {frame_path}")

def frames_to_frames(folder_path):
    '''Concatenate all TIFF files in a specified folder into a single 3D NumPy array.'''
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    print(f'total files: {tif_files}')
    tif_files.sort()
    
    images = []
    for tif_file in tif_files:
        file_path = os.path.join(folder_path, tif_file)
        img = tiff.imread(file_path)
        print(f'file: {file_path}\nstack shape: {img.shape}')
        images.append(img)

    concatenated_array = np.concatenate(images, axis=0)
    return concatenated_array

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
