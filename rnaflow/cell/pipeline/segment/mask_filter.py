import numpy as np

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