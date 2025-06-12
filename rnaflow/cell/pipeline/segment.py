import numpy as np
import cv2
import tifffile as tiff
from skimage.measure import regionprops

from mmdet.apis import init_detector, inference_detector
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# region Segmentor

def uint16_to_uint8_maxmin(uint16_array, max_int, min_int):
    offset = 100
    min_int = max(min_int-offset,0)
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

def check_distance(box_raw, img, threshold_overlap=0.8, threshold_distance=5,threshold_overlap_single=0.9):                  #0.5 /25/ 0.8 10.10change
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


def load_detector(device='cuda'):
    config_file = '/root/cell_track/bash/config.py'
    checkpoint_file = '/root/cell_track/bash/best_coco_bbox_mAP_epoch_180.pth'
    model = init_detector(config_file, checkpoint_file, device=device)
    return model

def load_segmentor(device='cuda'):
    sam_checkpoint = "/root/cell_track/bash/segment-anything/notebooks/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def array_to_rgb(array):
    bgr_array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return rgb_array

def run_detect(detector, img_rgb_path: str, img_rgb: np.ndarray, threshold: float, check_distance: bool = True):
    '''Run the detection model on the input image and filter bounding boxes based on a score threshold.'''
    res = inference_detector(detector, img_rgb_path)
    bbox_raw = res.pred_instances.bboxes
    score = res.pred_instances.scores
    if not threshold:
        return bbox_raw, score
    else:
        bbox_raw = bbox_raw.cpu().numpy()
        bbox_filter = []
        for i in range(bbox_raw.shape[0]):
            if score[i] > threshold:
                bbox_filter.append(bbox_raw[i])

        if check_distance:
            bbox_filter = check_distance(bbox_filter, img_rgb)
        bbox_filter = np.array(bbox_filter)
        if len(bbox_filter) == 0:
            raise ValueError("No valid bounding boxes found in the image.")
        return bbox_filter, score
    
def run_segment_with_bbox(predictor, img_pre: np.ndarray, bbox: np.ndarray):
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
    input_box = np.array(bbox)

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
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

def segment_fn(detector, predictor, img_rgb_path: str, img_pre_path: str,threshold: float = 0.1, check_distance: bool = True, bbox_last: list = None):
    img_pre = tiff.imread(img_pre_path)
    img_rgb = tiff.imread(img_rgb_path)
    bbox_filter, score = run_detect(detector, img_rgb_path, img_rgb, threshold, check_distance)
    predictor.set_image(img_rgb)
    
    bbox_save = []
    mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint16)
    for i in range(bbox_filter.shape[0]):
        bbox = bbox_filter[i]
        mask_segment = run_segment_with_bbox(predictor, img_pre, bbox)

        if mask_segment is not None:
            gray_index = i + 1
            mask[mask_segment] = gray_index
            bbox_save.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    
    return mask, bbox_save

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

def gray_to_rgb(img: np.ndarray, min_value: int, max_value: int) -> np.ndarray:
    if img.dtype == np.uint16:
        img = uint16_to_uint8_maxmin(img, min_value, max_value)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb_img

def segment_sr(img_rgb: np.ndarray, img_pre: np.ndarray, detector, sam_predictor, device: str) -> np.ndarray:
    pass
