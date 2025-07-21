import os
from os.path import join, basename
from multiprocessing import Pool, cpu_count
import numpy as np

# 仅保留SEG指标相关依赖
from ctc_metrics.metrics import seg
from ctc_metrics.utils.handle_results import print_results, store_results
from ctc_metrics.utils.filesystem import parse_directories, parse_masks
from ctc_metrics.utils.representations import match as match_tracks


# -------------------------- 路径配置 --------------------------
# 预测掩码文件夹路径（存放你的预测结果mask文件）
RES_PATH = r"C:\Users\wangfeihu\Desktop\test_dataset_ctc\test_dataset_ctc\train\BF-C2DL-HSC\01_RES"

# 真实标签根目录（需包含"SEG"子文件夹，SEG中存放真实mask文件）
GT_PATH = r"C:\Users\wangfeihu\Desktop\test_dataset_ctc\test_dataset_ctc\train\BF-C2DL-HSC\01_GT\SEG"

# 其他配置（可选）
RECURSIVE = False  # 是否递归评估子目录（批量评估多个序列时设为True）
CSV_FILE = None    # 结果保存路径（例如："./seg_results.csv"，不需要则设为None）
NUM_THREADS = 0    # 线程数（0表示使用所有CPU核心）
# ----------------------------------------------------------------------------------


def match_computed_to_reference_masks(
        ref_masks: list,
        comp_masks: list,
        threads: int = 0,
):
    """匹配预测掩码与真实掩码,计算IoU"""
    labels_ref, labels_comp, mapped_ref, mapped_comp, ious = [], [], [], [], []
    if threads != 1:
        if threads == 0:
            threads = cpu_count()
        with Pool(threads) as p:
            matches = p.starmap(match_tracks, zip(ref_masks, comp_masks))
    else:
        matches = [match_tracks(*x) for x in zip(ref_masks, comp_masks)]
    for match in matches:
        labels_ref.append(match[0])
        labels_comp.append(match[1])
        mapped_ref.append(match[2])
        mapped_comp.append(match[3])
        ious.append(match[4])
    return {
        "labels_ref": labels_ref,
        "labels_comp": labels_comp,
        "mapped_ref": mapped_ref,
        "mapped_comp": mapped_comp,
        "ious": ious
    }


def load_seg_data(
        res: str,
        gt: str,
        threads: int = 0,
):
    """加载SEG评测所需的掩码数据"""
    # 解析预测掩码和真实分割掩码
    comp_masks = parse_masks(res)
    ref_seg_masks = parse_masks(gt)  # 真实掩码固定在gt/SEG目录下
    
    # 校验数据有效性
    assert len(ref_seg_masks) > 0, f"错误：{gt}/SEG中未找到真实掩码文件！"
    assert len(ref_seg_masks) == len(comp_masks), (
        f"错误：预测掩码数量（{len(comp_masks)}）与真实掩码数量（{len(ref_seg_masks)}）不一致！"
    )
    
    # 匹配掩码并计算IoU
    segm = match_computed_to_reference_masks(
        ref_seg_masks, comp_masks, threads=threads
    )
    return segm


def calculate_seg_metric(segm: dict):
    """计算SEG指标"""
    return {"SEG": seg(segm["labels_ref"], segm["ious"])}


def evaluate_sequence(res: str, gt: str, threads: int = 0):
    """评估单个序列的SEG指标"""
    print(f"正在评估：预测路径={res}，真实路径={gt}")
    segm = load_seg_data(res, gt, threads=threads)
    results = calculate_seg_metric(segm)
    print(f"SEG结果：{results}\n")
    return results


def evaluate_all(res_root: str, gt_root: str, threads: int = 0):
    """批量评估多个序列的SEG指标(递归子目录)"""
    results = []
    # 解析所有子目录（需确保parse_directories能正确匹配预测和真实路径）
    ret = parse_directories(res_root, gt_root)
    for res, gt, name in zip(*ret):
        results.append([name, evaluate_sequence(res, gt, threads)])
    return results


def main():
    """主函数:执行SEG指标评测流程"""
    print("===== SEG(分割)指标评测 =====")
    # 根据配置选择单序列评估或批量评估
    if RECURSIVE:
        results = evaluate_all(RES_PATH, GT_PATH, threads=NUM_THREADS)
    else:
        results = [["single_sequence", evaluate_sequence(RES_PATH, GT_PATH, threads=NUM_THREADS)]]
    
    # 打印汇总结果
    print("\n===== 评测汇总 =====")
    print_results(results)
    
    # 保存结果到CSV（如果配置了路径）
    if CSV_FILE is not None:
        store_results(CSV_FILE, results)
        print(f"\n结果已保存至:{os.path.abspath(CSV_FILE)}")


if __name__ == "__main__":
    main()