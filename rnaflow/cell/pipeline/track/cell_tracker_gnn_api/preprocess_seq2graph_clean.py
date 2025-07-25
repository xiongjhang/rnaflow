import os
import os.path as op

from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from skimage import io
from skimage.measure import regionprops
import warnings
warnings.filterwarnings("always")

from models.metric_learning.modules.resnet import set_model_architecture, MLP
from skimage.morphology import label



class TestDataset(Dataset):
    """Example dataset class for loading images from folder."""

    def __init__(self,
                 path: str,
                 path_result: str,

                 type_img: str,
                 type_masks: str
                 ):

        self.path = path

        self.path_result = path_result

        dir_img = path
        dir_results = path_result

        self.images = []
        if os.path.exists(dir_img):
            self.images = [os.path.join(dir_img, fname) for fname in sorted(os.listdir(dir_img))
                           if type_img in fname]

        self.results = []
        if os.path.exists(dir_results):
            self.results = [os.path.join(dir_results, fname) for fname in sorted(os.listdir(dir_results))
                            if type_masks in fname]

    def __getitem__(self, idx):
        assert len(self.images) or len(self.images), "both directories are empty, please fix it!"

        im_path, image = None, None
        if len(self.images):
            im_path = self.images[idx]
            image = np.array(Image.open(im_path))  # .convert("L")  # convert to black and white

        result_path, result = None, None
        if len(self.results):
            result_path = self.results[idx]
            result = np.array(Image.open(result_path))
        flag = True
        if im_path is not None:
            flag = False
            im_num = im_path.split(".")[-2][-3:]

        if result_path is not None:
            flag = False
            result_num = result_path.split(".")[-2][-3:]

        if flag:
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

        return image, result, im_path, result_path

    def __len__(self):
        return len(self.images)

    def padding(self, img):
        if self.flag_new_roi:
            desired_size_row = self.global_delta_row
            desired_size_col = self.global_delta_col
        else:
            desired_size_row = self.roi_model['row']
            desired_size_col = self.roi_model['col']
        delta_row = desired_size_row - img.shape[0]
        delta_col = desired_size_col - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2

        image = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,
                                   cv2.BORDER_CONSTANT, value=self.pad_value)

        if self.flag_new_roi:
            image = cv2.resize(image, dsize=(self.roi_model['col'], self.roi_model['row']))

        return image

    def extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMaxCell'):
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb]
        msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        img_patch[msk_patch] = self.pad_value
        img_patch = img_patch.astype(np.float32)

        if normalize_type == 'regular':
            img = self.padding(img_patch) / self.max_img
        elif normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.min_cell) / (self.max_cell - self.min_cell)
            img = self.padding(img_patch)
        else:
            assert False, "Not supported this type of normalization"

        img = torch.from_numpy(img).float()
        with torch.no_grad():
            embedded_img = self.embedder(self.trunk(img[None, None, ...]))

        return embedded_img.numpy().squeeze()

    def correct_masks(self, min_cell_size):
        n_changes = 0
        for ind_data in range(self.__len__()):
            per_cell_change = False
            per_mask_change = False

            img, result, im_path, result_path = self[ind_data]
            res_save = result.copy()
            print(f"start: {result_path}")
            labels_mask = result.copy()
            while True:
                bin_mask = labels_mask > 0
                re_label_mask = label(bin_mask)
                un_labels, counts = np.unique(re_label_mask, return_counts=True)

                if np.any(counts < min_cell_size):
                    per_mask_change = True

                    # print(f"{im_path}: \n {counts}")
                    first_label_ind = np.argwhere(counts < min_cell_size)
                    if first_label_ind.size > 1:
                        first_label_ind = first_label_ind.squeeze()[0]
                    first_label_num = un_labels[first_label_ind]
                    labels_mask[re_label_mask == first_label_num] = 0
                else:
                    break
            bin_mask = (labels_mask > 0) * 1.0
            result = np.multiply(result, bin_mask)
            if not np.all(np.unique(result) == np.unique(res_save)):
                warnings.warn(
                    f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")

            # assert np.all(np.unique(result) == np.unique(res_save))
            for ind, id_res in enumerate(np.unique(result)):
                if id_res == 0:
                    continue
                bin_mask = (result == id_res).copy()
                while True:
                    re_label_mask = label(bin_mask)
                    un_labels, counts = np.unique(re_label_mask, return_counts=True)

                    if np.any(counts < min_cell_size):
                        per_cell_change = True
                        # print(f"{im_path}: \n {counts}")
                        first_label_ind = np.argwhere(counts < min_cell_size)
                        if first_label_ind.size > 1:
                            first_label_ind = first_label_ind.squeeze()[0]
                        first_label_num = un_labels[first_label_ind]
                        curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                        bin_mask[curr_mask] = False
                        result[curr_mask] = 0.0
                    else:
                        break
                while True:
                    re_label_mask = label(bin_mask)
                    un_labels, counts = np.unique(re_label_mask, return_counts=True)
                    if un_labels.shape[0] > 2:
                        per_cell_change = True
                        n_changes += 1
                        # print(f"un_labels.shape[0] > 2 : {im_path}: \n {counts}")
                        first_label_ind = np.argmin(counts)
                        if first_label_ind.size > 1:
                            first_label_ind = first_label_ind.squeeze()[0]
                        first_label_num = un_labels[first_label_ind]
                        curr_mask = np.logical_and(result == id_res, re_label_mask == first_label_num)
                        bin_mask[curr_mask] = False
                        result[curr_mask] = 0.0
                    else:
                        break
            if not np.all(np.unique(result) == np.unique(res_save)):
                warnings.warn(
                    f"pay attention! the labels have changed from {np.unique(res_save)} to {np.unique(result)}")
            if per_cell_change or per_mask_change:
                n_changes += 1
                res1 = (res_save > 0) * 1.0
                res2 = (result > 0) * 1.0
                n_pixels = np.abs(res1 - res2).sum()
                print(f"per_mask_change={per_mask_change}, per_cell_change={per_cell_change}, number of changed pixels: {n_pixels}")
                stp = 2
                io.imsave(result_path, result.astype(np.uint16), compress=6)
            # assert np.all(np.unique(result) == np.unique(res_save))


        print(f"number of detected changes: {n_changes}")

        # patch = img[min_row_bb: max_row_bb, min_col_bb: max_col_bb]
        # fig, ax = plt.subplots(1, 2, figsize=(17, 6))
        # ax[0].imshow(patch, cmap='gray')
        # ax[1].imshow(result[min_row_bb: max_row_bb, min_col_bb: max_col_bb] == id_res, cmap='gray')
        # plt.show()

    def find_min_max_and_roi(self):
        global_min = 2 ** 16 - 1
        global_max = 0
        global_delta_row = 0
        global_delta_col = 0
        counter = 0
        for ind_data in range(self.__len__()):
            img, result, im_path, result_path = self[ind_data]

            for ind, id_res in enumerate(np.unique(result)):
                if id_res == 0:
                    continue

                properties = regionprops(np.uint8(result == id_res), img)[0]
                min_row_bb, min_col_bb, max_row_bb, max_col_bb = properties.bbox
                delta_row = np.abs(max_row_bb - min_row_bb)
                delta_col = np.abs(max_col_bb - min_col_bb)

                if (delta_row > self.roi_model['row']) or (delta_col > self.roi_model['col']):
                    counter += 1

                global_delta_row = max(global_delta_row, delta_row)
                global_delta_col = max(global_delta_col, delta_col)

            res_bin = result != 0
            min_curr = img[res_bin].min()
            max_curr = img[res_bin].max()

            global_min = min(global_min, min_curr)
            global_max = max(global_max, max_curr)
        print(counter)
        print(f"global_delta_row: {global_delta_row}")
        print(f"global_delta_col: {global_delta_col}")
        self.min_cell = global_min
        self.max_cell = global_max

        self.global_delta_row = global_delta_row
        self.global_delta_col = global_delta_col

    def preprocess_features_loop_by_results_w_metric_learning(self, path_to_write, dict_path):
        dict_params = torch.load(dict_path)

        self.roi_model = dict_params['roi']
        self.find_min_max_and_roi()
        self.flag_new_roi = self.global_delta_row > self.roi_model['row'] or self.global_delta_col > self.roi_model['col']
        if self.flag_new_roi:
            self.global_delta_row = max(self.global_delta_row, self.roi_model['row'])
            self.global_delta_col = max(self.global_delta_col, self.roi_model['col'])
            print("Assign new region of interest")
            print(f"old ROI: {self.roi_model}, new: row: {self.global_delta_row}, col : {self.global_delta_col}")
        else:
            print("We don't assign new region of interest - use the old one")

        self.pad_value = dict_params['pad_value']
        # models params
        model_name = dict_params['model_name']
        mlp_dims = dict_params['mlp_dims']
        mlp_normalized_features = dict_params['mlp_normalized_features']
        # models state_dict
        trunk_state_dict = dict_params['trunk_state_dict']
        embedder_state_dict = dict_params['embedder_state_dict']

        trunk = set_model_architecture(model_name)
        trunk.load_state_dict(trunk_state_dict)
        self.trunk = trunk
        self.trunk.eval()

        embedder = MLP(mlp_dims, normalized_feat=mlp_normalized_features)
        embedder.load_state_dict(embedder_state_dict)
        self.embedder = embedder
        self.embedder.eval()

        # id, area, bbox_area, min_row_bb, min_col_bb, max_row_bb, max_col_bb, centroid_row, centroid_col = 8 + id = 9
        # max_intensity, mean_intensity, min_intensity, orientation, perimeter, weighted_centroid_row, weighted_centroid_col = 7
        # equivalent_diameter: float = The diameter of a circle with the same area as the region. ???

        cols = ["seg_label",
                "frame_num",
                "area",
                "bbox_area",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "major_axis_length", "minor_axis_length",
                "max_intensity", "mean_intensity", "min_intensity",
                "orientation", "perimeter",
                "weighted_centroid_row", "weighted_centroid_col"
                ]


        cols_resnet = [f'feat_{i}' for i in range(mlp_dims[-1])]
        cols += cols_resnet

        for ind_data in range(self.__len__()):
            img, result, im_path, result_path = self[ind_data]

            im_num = im_path.split(".")[-2][-3:]
            result_num = result_path.split(".")[-2][-3:]
            assert im_num == result_num, f"Image number ({im_num}) is not equal to result number ({result_num})"

            num_labels = np.unique(result).shape[0] - 1

            df = pd.DataFrame(index=range(num_labels), columns=cols)

            for ind, id_res in enumerate(np.unique(result)):
                # Color 0 is assumed to be background or artifacts
                row_ind = ind - 1
                if id_res == 0:
                    continue

                # extracting statistics using regionprops
                properties = regionprops(np.uint8(result == id_res), img)[0]

                embedded_feat = self.extract_freature_metric_learning(properties.bbox, img.copy(), result.copy(), id_res)
                df.loc[row_ind, cols_resnet] = embedded_feat
                df.loc[row_ind, "seg_label"] = id_res

                df.loc[row_ind, "area"], df.loc[row_ind, "bbox_area"] = properties.area, properties.bbox_area

                df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
                df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox

                df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
                    properties.centroid[0].round().astype(np.int16), \
                    properties.centroid[1].round().astype(np.int16)

                df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
                    properties.major_axis_length, properties.minor_axis_length

                df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
                    properties.max_intensity, properties.mean_intensity, properties.min_intensity

                df.loc[row_ind, "orientation"], df.loc[row_ind, "perimeter"] = properties.orientation, \
                                                                               properties.perimeter
                if properties.weighted_centroid[0] != properties.weighted_centroid[0] or properties.weighted_centroid[
                    1] != properties.weighted_centroid[1]:
                    df.loc[row_ind, "weighted_centroid_row"], df.loc[
                        row_ind, "weighted_centroid_col"] = properties.centroid
                else:
                    df.loc[row_ind, "weighted_centroid_row"], df.loc[
                        row_ind, "weighted_centroid_col"] = properties.weighted_centroid

            df.loc[:, "frame_num"] = int(im_num)

            if df.isnull().values.any():
                warnings.warn("Pay Attention! there are Nan values!")

            full_dir = op.join(path_to_write, "csv")
            os.makedirs(full_dir, exist_ok=True)
            file_path = op.join(full_dir, f"frame_{im_num}.csv")
            print(f"save file to : {file_path}")
            df.to_csv(file_path, index=False)


def create_csv(input_images, input_seg, input_model, output_csv, min_cell_size):
    dict_path = input_model
    path_output = output_csv
    path_Seg_result = input_seg
    ds = TestDataset(
        path=input_images,
        path_result=path_Seg_result,
        type_img="tif",
        type_masks="tif")
    ds.correct_masks(min_cell_size)
    ds.preprocess_features_loop_by_results_w_metric_learning(path_to_write=path_output,
        dict_path=dict_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-ii', type=str, required=True, help='input images directory')
    parser.add_argument('-iseg', type=str, required=True, help='input segmentation directory')
    parser.add_argument('-im', type=str, required=True, help='metric learning model params directory')
    parser.add_argument('-cs', type=int, required=True, help='min cell size')

    parser.add_argument('-oc', type=str, required=True, help='output csv directory')

    args = parser.parse_args()

    min_cell_size = args.cs
    input_images = args.ii
    input_segmentation = args.iseg
    input_model = args.im

    output_csv = args.oc

    create_csv(input_images, input_segmentation, input_model, output_csv, min_cell_size)
