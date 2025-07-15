'''Sunrui version of segmentation using sam and detector.'''
import sys
from typing import Any, Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import logging

import torch
import cv2
import numpy as np
import tifffile as tiff
from skimage.measure import regionprops

# from mmdet.apis import init_detector, inference_detector
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from rnaflow.cell.pipeline.utils import get_logger
from rnaflow.cell.pipeline.segment.mask_filter import correct_mask_fast

logger = get_logger(__name__, level=logging.INFO)

SAM_CHECKPOINT = {
    "vit_b": "/mnt/sda/xjh/pt/sam/sam_vit_b_01ec64.pth",
    "vit_l": "/mnt/sda/xjh/pt/sam/sam_vit_l_0b3195.pth",
    "vit_h": "/mnt/sda/xjh/pt/sam/sam_vit_h_4b8939.pth",
}

YOLO_WEIGHT_CONFIG = {
    "yolov5x": {
        "Fluo-N2DH-SIM+": "/mnt/sda/cell_data_xjh/clx/experiments/train/yolov5x6u_Fluo-N2DH-SIM+4/weights/best.pt",
        "Fluo-N2DL-HeLa": "/mnt/sda/cell_data_xjh/clx/experiments/train/yolov5x6u_Fluo-N2DL-HeLa/weights/best.pt",
        "sunrui": "/mnt/sda/cell_data_xjh/clx/experiments/train/yolov5x6u_sunri/weights/best.pt",
        "Hybrid": "/mnt/sda/cell_data_xjh/clx/experiments/train/yolov5x6u_all_dataset/weights/best.pt",
    },
    "yolov8x": {
        "Fluo-N2DH-SIM+": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_Fluo_N2DH-SIM+/weights/best.pt",
        "Fluo-N2DL-HeLa": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_Fluo_N2DL-HeLa/weights/best.pt",
        "sunrui": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_sunri/weights/best.pt",
        "Hybrid": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov8x_all_dataset/weights/best.pt",
    },
    "yolov11x": {
        "Fluo-N2DH-SIM+": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_Fluo_N2DH-SIM+",
        "Fluo-N2DL-HeLa": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_Fluo_N2DL-HeLa",
        "sunrui": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_sunri/weights/best.pt",
        "Hybrid": "/mnt/sda/cell_data_xjh/wfh/shiyan/runs/yolov11x_all_dataset/weights/best.pt",
    },
}

MMDET_CONFIG = {
    "yolov8": {
        "config": "/root/cell_track/bash/config.py",
        "checkpoint": "/root/cell_track/bash/best_coco_bbox_mAP_epoch_180.pth",
    }
}

def prepare(
        det_exp_name,
        detector_type: str = "mmdet",
        detector_name: str = "yolov8",
        sam_model_type: str = "vit_h",
        device: str = "cuda",
):
    # seg model
    logger.info(f"Loading sam model...")
    sam = sam_model_registry[sam_model_type](checkpoint=SAM_CHECKPOINT[sam_model_type])
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # dect model
    logger.info(f"Loading detector model...")
    if detector_type == "mmdet":
        config = MMDET_CONFIG[detector_name]["config"]
        checkpoint = MMDET_CONFIG[detector_name]["checkpoint"]
        # detector = init_detector(config, checkpoint, device=device)
        detector = None
    elif detector_type == "yolo":
        checkpoint = YOLO_WEIGHT_CONFIG[detector_name][det_exp_name]
        detector = YOLO(checkpoint)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}. Supported types are 'mmdet' and 'yolo'.")

    return sam_predictor, detector  

def run_detect(
        detector, 
        detector_type: str,  
        img_rgb_path: Path, 
        threshold: float, 
        check_box_iou: bool = True,
        save_annotated: bool = False
):
    '''Run the detection model on the input image and filter bounding boxes based on a score threshold.'''
    img_rgb = tiff.imread(img_rgb_path)
    assert img_rgb.ndim == 3, f"Input image must be a 3D array, got shape {img_rgb.shape}"

    # run detection
    # TODO: plot box on raw image
    if detector_type == "mmdet":
        # res = inference_detector(detector, img_rgb_path)
        res = None
        bbox_raw = res.pred_instances.bboxes  # TODO: which type
        score = res.pred_instances.scores  # TODO: which type
        bbox_raw = bbox_raw.cpu().numpy()

    elif detector_type == "yolo":
        results = detector(img_rgb, conf = threshold)
        boxes = results[0].boxes
        if isinstance(boxes.xywh, torch.Tensor):
            bbox_raw = boxes.xywh.cpu().numpy()
            score = boxes.conf.cpu().numpy()
        else:
            bbox_raw = boxes.xywh
            score = boxes.conf
        bbox_raw = xywh2xyxy(bbox_raw)

        if save_annotated:
            annotated_image = results[0].plot()
            annotated_image_path = img_rgb_path.parent / f"annotated_{img_rgb_path.name}"
            cv2.imwrite(annotated_image_path, annotated_image)

    if not threshold:
        return bbox_raw, score
    else:
        bbox_filter = []
        for i in range(bbox_raw.shape[0]):
            if score[i] > threshold:
                bbox_filter.append(bbox_raw[i])

        if check_box_iou:
            bbox_filter = check_distance(bbox_filter, img_rgb)

        if len(bbox_filter) == 0:
            raise ValueError("No valid bounding boxes found in the image.")
        
        return np.array(bbox_filter), score

def run_segment_with_bbox(
        predictor, 
        img_pre: np.ndarray, 
        bbox: np.ndarray
):  
    '''Run the SAM model to segment the image using a bounding box.'''
    assert img_pre.ndim == 2, f"Input image must be a 2D array, got shape {img_pre.shape}"
    assert isinstance(bbox, np.ndarray), f"bbox must be a numpy array, got {type(bbox)}"

    # point prompt from bbox
    x0, y0 = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = x0 + w/2
    y = y0 + h/2
    h_img = img_pre.shape[0]-1
    w_img = img_pre.shape[1]-1
    x = min(x,w_img)
    y = min(y,h_img)
    input_point = np.array([[x, y]])
    input_label = np.array([1])

    int_cent = img_pre[int(y)][int(x)].sum()
    if int_cent <= 3:
        return None

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=False,
    )
    if scores[0] < 0.05:
        return None
    num_pixels = np.sum(masks[0])
    thresholdU = 20000000000000                  
    thresholdD = 6                                       
    if num_pixels > thresholdU or num_pixels < thresholdD :
        return None
    return masks[0]

# Pure inference function for segmenting images
def seg_per_frame(
        detector,
        detector_type,
        predictor,
        img_path: Path,
        img_rgb_path: Path,
        using_prompt: bool = False,
        box_prompt: Optional[np.ndarray] = None,
        threshold: float = 0.1,
        check_distance: bool = True
):
    """Segment an image using the SAM model with a detector.

    Args:
        detector: the detector model to use for bounding box detection.
        predictor: the SAM predictor model to use for segmentation.
        img_path: path to the input image.
        img_rgb_path: path to the RGB version of the input image.
        using_prompt: whether to use a prompt for segmentation.
        box_prompt: optional bounding box prompt to use for segmentation.
        threshold: score threshold for filtering bounding boxes from the detector.
        check_distance: whether to check the distance between bounding boxes.
        
    Returns:
        mask: the segmented mask of the image.
        bbox_save: list of bounding boxes used for segmentation.
    """
    img = tiff.imread(img_path)
    # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert img.ndim == 2, f"Input image must be a 2D array, got shape {img.shape}"

    if not img_rgb_path.exists():
        # convert to rgb and save
        assert img.dtype == np.uint16, f"Input image must be a uint16 array, got dtype {img.dtype}"
        img_rgb = gray_to_rgb(img, min_value=np.min(img), max_value=np.max(img))
        tiff.imwrite(img_rgb_path, img_rgb)
    else:
        img_rgb = tiff.imread(img_rgb_path)

    # which bbox to use
    if using_prompt:
        if box_prompt is not None:
            box_input = box_prompt
        else:
            raise ValueError("Using prompt but no box_prompt provided.")
    else:
        # run detection
        bbox_filter, score = run_detect(detector, detector_type, img_rgb_path, threshold, check_distance)
        box_input = bbox_filter
    assert isinstance(box_input, np.ndarray), f"box_input must be a numpy array, got {type(box_input)}"

    # run segmentation
    predictor.set_image(img_rgb)
    
    bbox_save = []
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint16)
    for i in range(box_input.shape[0]):
        bbox = bbox_filter[i]
        mask_segment = run_segment_with_bbox(predictor, img, bbox)

        if mask_segment is not None:
            gray_index = i + 1
            mask[mask_segment] = gray_index

            # Which prompt used from previous segmentations?
            # version 1: save box prompt
            # bbox_save.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            # version 2: save box from mask
            bbox_save.append(mask_to_bbox(mask_segment.astype(np.uint8)))
    
    return mask, bbox_save

# Pipeline function to segment images and save results
def segment_fn(
        input: Union[Path, List[Path]],
        save_raw_dir: Path,
        save_curated_dir: Path,
        device: str = 'cuda',
        prefix: str = 'frame_',
        batch_size: Optional[int] = None,
        detector_type: str = "mmdet",
        detector_name: str = "yolov8",
        det_exp_name: str = "hybrid",
        sam_model_type: str = "vit_h",
        using_pre_frame_prompt: bool = False,
):
    """Segment an image using the Cellpose model."""
    # TODO: support batch inference

    if isinstance(input, Path):
        input =  sorted(input.glob('*.tif'))
    img_paths = input
    
    # get model
    segmentor, detector = prepare(
        det_exp_name=det_exp_name,
        detector_type=detector_type,
        detector_name=detector_name,
        sam_model_type=sam_model_type,
        device=device,
    )
    
    # model inference
    bbox_input = None   # first frame only have detector bbox as prompt
    for idx, img_path in tqdm(enumerate(img_paths), desc='Segmenting images', unit='image'):
        img_pre_dir = img_path.parent
        img_rgb_path = img_pre_dir / 'RGB' / img_path.name
        img_rgb_path.parent.mkdir(parents=True, exist_ok=True)

        mask, bbox_save = seg_per_frame(
                    detector,
                    detector_type,
                    segmentor, 
                    img_path,
                    img_rgb_path,
                    using_pre_frame_prompt,
                    bbox_input,
                )
        bbox_input = bbox_save  # use the bbox from the previous frame as prompt for the next frame
        
        # mask = mask[0]  # get the first mask, assuming single image input
        dst_img_path = save_raw_dir / f'{prefix}{idx:04d}.tif'
        tiff.imwrite(str(dst_img_path), mask)
        
        # # correct mask
        # dst_img_path = save_curated_dir / f'{prefix}filter_{idx:04d}.tif'
        # empty_img = np.ones(shape=(1, 1))
        # mask = correct_mask_fast(empty_img, mask, dst_img_path)


# region utils

# Official sam predictor api
def _sam_predict(
        segmentor: SamPredictor,
        img: Union[Path, np.ndarray],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
):
    """Segment an image using the SAM model.
    Args:
        img: image_format must be in ['RGB', 'BGR']
    """
    if isinstance(img, Path):
        img = tiff.imread(img)
    assert isinstance(img, np.ndarray), "Input image must be a numpy array or a path to an image file."

    segmentor.set_image(img)
    masks, scores, logits = segmentor.predict(
        point_coords = point_coords,
        point_labels = point_labels,
        box = box,
        mask_input = mask_input,
        multimask_output = multimask_output,
        return_logits = return_logits,
    )
    return masks

def uint16_to_uint8_maxmin(uint16_array, max_int, min_int):
    offset = 100
    min_int = min_int - offset if min_int >= offset else 0
    min_int = max(min_int-offset,0)
    max_int = max_int + offset if max_int <= 65535 - offset else 65535
    max_int = min(65535,max_int+offset)
    # print('min_int',min_int)
    # print('max_int',max_int)
    # 避免除以零的情况
    range_int = max_int - min_int
    if range_int == 0:
        range_int = 1  # 设置一个合理的默认值
    
    # 归一化到 [0, 1] 范围内
    normalized_array = (uint16_array - min_int) / range_int
    
    # 如果需要加偏移量，请明确这样做，并考虑其对结果的影响
    
    # 这里我们不加偏移量，直接缩放到 [0, 255]
    scaled_array = normalized_array * 255
    
    # 确保所有值都在 [0, 255] 范围内
    uint8_array = np.clip(scaled_array, 0, 255).astype(np.uint8)
    
    return uint8_array

def array_to_rgb(array):
    bgr_array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return rgb_array

def gray_to_rgb(img: np.ndarray, min_value: int = 0, max_value: int = 65535) -> np.ndarray:
    if img.dtype == np.uint16:
        img = uint16_to_uint8_maxmin(img, max_value, min_value)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb_img

def mask_to_bbox(mask):
    # 检测边界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # bboxes = []
    for contour in contours:
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 转换为标准边界框格式 [x1, y1, x2, y2]
        bbox = [x, y, x + w, y + h]
        
        # bboxes.append(bbox)

    return bbox

def xywh2xyxy(bboxes):
    """
    将边界框从 xywh（中心坐标+宽高）格式转换为 x1y1x2y2（左上角+右下角）格式。
    
    参数:
        bboxes: 形状为 [N, 4] 的 NumPy 数组，每行包含 [x, y, w, h]
        
    返回:
        形状为 [N, 4] 的 NumPy 数组，每行包含 [x1, y1, x2, y2]
    """
    # 创建一个新数组存储转换后的边界框
    bboxes_xyxy = np.zeros_like(bboxes)
    
    # 提取各列
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # 转换计算
    bboxes_xyxy[:, 0] = x - w / 2  # x1
    bboxes_xyxy[:, 1] = y - h / 2  # y1
    bboxes_xyxy[:, 2] = x + w / 2  # x2
    bboxes_xyxy[:, 3] = y + h / 2  # y2
    
    return bboxes_xyxy

def check_distance(box_raw, img, threshold_overlap=0.8, threshold_distance=5,threshold_overlap_single=0.9):                  #0.5 /25/ 0.8 10.10change
    """Sunrui filter method based on box"""
    # 初始化列表以存储筛选后的边界框
    bbx_save = []
    area_all = []
    xy_all = np.zeros((2, len(box_raw)))  

    # box_raw = box_raw.detach().numpy()  # 如果是PyTorch的Tensor
    # 或者
    box_raw = np.array(box_raw) # 如果是TensorFlow的Tensor

    
    # 遍历每个边界框
    for i, ids in enumerate(box_raw):
        area = 0
        box = ids
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        
        # 计算面积以备后用
        area = w * h
        
        if area<250:
            continue

        area_all.append(area)  
        
        # 计算边界框的中心点
        x = x0 + w / 2
        y = y0 + h / 2
        
        # 确保中心点在图像范围内
        h_img, w_img = img.shape[:2]
        x = min(x, w_img)
        y = min(y, h_img)
        
        # 存储中心点坐标
        xy_all[0, i] = x
        xy_all[1, i] = y
        
        # 检查边界框之间是否存在重叠,
        overlap = False
        for bbx in bbx_save:
            # 计算交集面积
            x_left = max(bbx[0], x0)
            y_top = max(bbx[1], y0)
            x_right = min(bbx[2], box[2])
            y_bottom = min(bbx[3], box[3])


            bbx_x = bbx[0] + (bbx[2] - bbx[0])/2
            bbx_y = bbx[1] + (bbx[3] - bbx[1])/2


            distance = distance = np.sqrt((x - bbx_x)**2 + (y - bbx_y)**2)
            if distance < threshold_distance:
                # 判断是否为嵌套关系，是否全为内部，是则保留嵌套中小的mask
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)#相交面积
                    bbx_area = (bbx[2] - bbx[0])*(bbx[3] - bbx[1])#bbx面积
                    iou = intersection_area / (area + bbx_area - intersection_area)#两者交并比
                    rate_1 = intersection_area/bbx_area #交集比单个mask（bbx）
                    rate_2 = intersection_area/area #交集比单个mask（bbox）
                    area_1 = bbx_area
                    area_2 = area
                    
                    if rate_2 > threshold_overlap_single and bbx_area > area:
                        # print('rate_2:',rate_2)
                        # print('rate_1:',rate_1)
                        if (bbx_save == bbx).all():  # 使用 .all() 方法检查数组中的所有元素是否与当前边界框匹配
                            bbx_save.remove(bbx)
                        # bbx_save.remove(bbx)
                    elif rate_1 > threshold_overlap_single and bbx_area < area:
                        # print('rate_2:',rate_2)
                        # print('rate_1:',rate_1)
                        overlap = True
                        break
        
        # 如果边界框通过了以上两个条件，则将其添加到保存的筛选后边界框列表中
        if not overlap :
            bbx_save.append(ids)
    
    return bbx_save


# region Segmentation Post-processing

def check_mask_intensity(image_t,mask_t,intensity_t):
    
    if intensity_t == 0:
        return False
    
    properties = regionprops(np.uint16(mask_t == 255), mask_t)[0]

    center_x = properties.centroid[1].round().astype(np.int16)
    center_y = properties.centroid[0].round().astype(np.int16)
    # 找到掩模中的非零点
    # nonzero_points = np.column_stack(np.where(mask_t > 0))

    # if len(nonzero_points) == 0:

    #     return False

    # # 计算掩模中心
    # center_x = int(np.mean(nonzero_points[:, 1]))
    # center_y = int(np.mean(nonzero_points[:, 0]))

    # 提取中心附近的像素强度值
    radius = 2  # 设置半径为10像素，你可以根据需要调整
    intensity_values = []
    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            # 确保不超出图像范围
            if 0 <= x < image_t.shape[1] and 0 <= y < image_t.shape[0]:
                intensity_values.append(image_t[y, x])
    
    # 判断中心附近的强度值是否与掩模中心的值相同
    center_intensity = intensity_t
    are_values_same = all(value == center_intensity for value in intensity_values)
    if not are_values_same:
        # plt.figure(figsize=(25, 25))
        # plt.subplot(1,2,1)
        # print('find a hole!')
        # print('intensity',intensity_t)
        # print(intensity_values)
        # print(center_x,center_y)
        # plt.scatter(center_x, center_y, color='red', marker='*', label='Center Point')
        
        # plt.imshow(mask_t)

        # plt.subplot(1,2,2)
        # plt.imshow(image_t)
        # plt.show()

        # x = 1

        pass
    return are_values_same

def segment_postprocess_fn(mask: np.ndarray):
    min_value = np.min(mask)
    max_value = np.max(mask)

    mask_post = np.zeros_like(mask, dtype=np.uint16)
    for idx, value in enumerate(np.unique(mask)):
        if value == 0:
            continue
        gray_value_image = (mask == value).astype('uint8') * 255

        # 定义结构元素（这里使用3x3的正方形结构元素）
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 进行腐蚀操作
        eroded_image_0 = cv2.erode(gray_value_image, structuring_element)
        eroded_image_1 = cv2.erode(eroded_image_0, structuring_element)
        # 进行膨胀操作
        dilated_image_0 = cv2.dilate(eroded_image_1, structuring_element)
        dilated_image_1 = cv2.dilate(dilated_image_0, structuring_element)
        # 计算非零像素的数量（PRE）
        non_zero_pixels_pre = np.count_nonzero(gray_value_image)
        # 计算非零像素的数量（POST）
        non_zero_pixels_post = np.count_nonzero(dilated_image_1)

        post_result = (dilated_image_1 / 255) * value
        post_result = post_result.astype('uint16')
        ss = np.max(post_result)

        if non_zero_pixels_post > 230:
            # print('frame_ID:',i )
            is_same = check_mask_intensity(mask, gray_value_image, value)
            
            if is_same:
                mask_post += post_result

    return mask_post

''' Raw code:

pre_img0 = tiff.imread(self.pre_imgs_paths[0])
min_value0 = np.min(pre_img0)
max_value0 = np.max(pre_img0)

# convert preprocessed images to RGB format for segmentation
logger.info('Converting preprocessed images to RGB format...')
self.rgb_data_dir = self.dst_dir / 'PRE' / 'RGBGT'; self.rgb_data_dir.mkdir(parents=True, exist_ok=True)
_ = map_fn_to_frames(self.pre_imgs_paths, gray_to_rgb, save_dir=self.rgb_data_dir, save_prefix='test_',
                        max_value=max_value0, min_value=min_value0)

# for idx, img_path in enumerate(self.pre_imgs_paths):
#     img = tiff.imread(img_path)
#     if img.dtype == np.uint16:
#         img = uint16_to_uint8_maxmin(img, max_value0, min_value0)

#     bgr_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     dst_img_path = self.rgb_data_dir / f'test_{idx:04d}.tif'
#     tiff.imwrite(dst_img_path, rgb_image)

# init segmentation
self.rgb_imgs_paths = sorted(glob.glob(str(self.rgb_data_dir / '*.tif')))
self.seg_data_dir = self.gt_data_dir / 'SAMSEG'; self.seg_data_dir.mkdir(parents=True, exist_ok=True)
detector = load_detector(self.device)
sam_predictor = load_segmentor(self.device)

# segment first frame
rgb_img0_path = self.rgb_imgs_paths[0]
pre_img0_path = self.pre_imgs_paths[0]
mask0, bbox_save = segment_fn(detector, sam_predictor, rgb_img0_path, pre_img0_path)
dst_path = self.seg_data_dir / f'man_seg{0:04d}.tif'
tiff.imwrite(dst_path, mask0)

# segment other frames
for frame, img_pre_path, img_rgb_path in enumerate(zip(self.pre_imgs_paths, self.rgb_imgs_paths)):
    if frame == 0:
        continue
    
    # TODO: use the saved bbox from the first frame
    mask, _ = segment_fn(detector, sam_predictor, img_rgb_path, img_pre_path, bbox_save)
    dst_img_path = self.seg_data_dir / f'man_seg{frame:04d}.tif'
    tiff.imwrite(dst_img_path, mask)

# segmentation post-processing
self.seg_imgs_paths = sorted(glob.glob(str(self.seg_data_dir / '*.tif'))) 
self.seg_post_data_dir = self.gt_data_dir / 'SEG'; self.seg_post_data_dir.mkdir(parents=True, exist_ok=True)  # SEG_16

for frame, mask_path in enumerate(self.seg_imgs_paths):
    mask = tiff.imread(mask_path).astype(np.uint16)
    # Apply post-processing to the mask
    # TODO: rename - when the mask is empty, it will cause an error
    mask_post = segment_postprocess_fn(mask)

    dst_img_path = self.seg_post_data_dir / f'man_seg{frame:04d}.tif'
    tiff.imwrite(dst_img_path, mask_post)
'''