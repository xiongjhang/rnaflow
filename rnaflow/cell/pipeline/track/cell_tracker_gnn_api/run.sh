#!/bin/bash

# activate environment
source activate cell-tracking-challenge

# Inference script for cell tracking using Cell Tracker GNN from
# /home/xiongjiahang/repo/cell-tracker-gnn-software

SEQUENCE="01"
# SPACING="999 0.125 0.125"
# FOV="0"
# SIZES="15 10000"
DATASET="/mnt/sda/xjh/dataset/cell-data/test_pipeline/Fluo-N2DH-SIM+/cell-track-gnn"   # change this to your dataset path
INPUT_SEG="${SEQUENCE}_SEG_RES"
TRACK_DIR="${DATASET}/${SEQUENCE}_TRACK/cell_tracker_gnn"

CODE_TRA="${PWD}"
# CODE_SEG="${PWD}/seg_code/old_version"
# SEG_MODEL="${PWD}/parameters/Seg_Models/Fluo-N2DH-SIM+/"

MODEL_METRIC_LEARNING="/home/xiongjiahang/repo/cell-tracker-gnn-software/parameters/Features_Models/Fluo-N2DH-SIM+/all_params.pth"
MODEL_PYTORCH_LIGHTNING="/home/xiongjiahang/repo/cell-tracker-gnn-software/parameters/Tracking_Models/Fluo-N2DH-SIM+/checkpoints/epoch=132.ckpt"
MODALITY="2D"
        
# seg prediction
# python ${CODE_SEG}/Inference2D.py --gpu_id 0 --model_path ${SEG_MODEL} --sequence_path "${DATASET}/${SEQUENCE}" --output_path "${DATASET}/${SEQUENCE}_SEG_RES" --edge_dist 3 --edge_thresh=0.3 --min_cell_size 100 --max_cell_size 1000000 --fov 0 --centers_sigmoid_threshold 0.8 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ${DATASET}/${SEQUENCE}_SEG_intermediate

# cleanup
# rm -r "${DATASET}/${SEQUENCE}_SEG_intermediate"

# Finish segmentation - start tracking

# our model needs CSVs, so let's create from image and segmentation.
echo "Preprocessing sequence ${SEQUENCE} for tracking..."
python ${CODE_TRA}/preprocess_seq2graph_clean.py -cs 20 -ii "${DATASET}/${SEQUENCE}" -iseg "${DATASET}/${INPUT_SEG}" -im "${MODEL_METRIC_LEARNING}" -oc "${TRACK_DIR}/${SEQUENCE}_CSV"

# run the prediction
echo "Running inference for sequence ${SEQUENCE}..."
python ${CODE_TRA}/inference_clean.py -mp "${MODEL_PYTORCH_LIGHTNING}" -ns "${SEQUENCE}" -oc "${TRACK_DIR}"

# postprocess
echo "Postprocessing sequence ${SEQUENCE}..."
python ${CODE_TRA}/postprocess_clean.py -modality "${MODALITY}" -iseg "${DATASET}/${INPUT_SEG}" -oi "${TRACK_DIR}/${SEQUENCE}_RES_inference"

# rm -r "${DATASET}/${SEQUENCE}_CSV" "${DATASET}/${SEQUENCE}_RES_inference" "${DATASET}/${SEQUENCE}_SEG_RES"