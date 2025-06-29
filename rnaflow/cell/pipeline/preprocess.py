import numpy as np

'''
sunrui version:

    for idx, img_path in enumerate(self.raw_imgs_paths):
        img = tiff.imread(img_path)
        img = hist_match(img, img0)
        img = uint8_to_uint16(img)
        dst_img_path = self.pre_data_dir / f'test_{idx:04d}.tif'
        tiff.imwrite(dst_img_path, img)
'''

def hist_match(source, template):  
    # 计算源图像和目标图像的直方图  
    source_hist, _ = np.histogram(source.ravel(), 256, [0, 256])  
    template_hist, _ = np.histogram(template.ravel(), 256, [0, 256])  
  
    # 归一化直方图  
    source_hist = source_hist.astype('float')  
    template_hist = template_hist.astype('float')  
    source_hist /= (source_hist.sum() + 1e-7)  # 避免除以零  
    template_hist /= (template_hist.sum() + 1e-7)  
  
    # 计算累积分布函数（CDF）  
    source_cdf = source_hist.cumsum()  
    template_cdf = template_hist.cumsum()  
  
    # 创建映射表  
    mapping = np.zeros(256)  
    for i in range(256):  
        # 找到最接近的累积分布值  
        diff = template_cdf - source_cdf[i]  
        idx = np.argmin(np.abs(diff))  
        mapping[i] = idx  
  
    # 应用映射表到源图像  
    matched = np.interp(source.ravel(), np.arange(256), mapping)  
    matched = matched.reshape(source.shape)  
  
    return matched.astype('uint8')  

def uint8_to_uint16(uint8_array):
    # 将float32的值限制在0到1之间
    # float_value = np.clip(float_value, 0.0, 1.0)
    
    # # 将float32乘以255，并进行舍入
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
        
    scaled_array = (uint8_array.astype(np.float32) / 255.0) * 65535.0
    uint16_array = scaled_array.astype(np.uint16)

    return uint16_array

def preprocess_sr(img: np.ndarray, img_ref: np.ndarray) -> np.ndarray:
    '''Refactor by sunrui version'''
    assert img.ndim == 2, f"Input image must be a 2D array, got shape {img.shape}"
    assert img_ref.ndim == 2, f"Reference image must be a 2D array, got shape {img_ref.shape}"
    img = hist_match(img, img_ref)
    img = uint8_to_uint16(img)
    return img