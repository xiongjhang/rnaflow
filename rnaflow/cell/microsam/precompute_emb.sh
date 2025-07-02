#!/bin/bash

INPUT_PATH="/mnt/sda/xjh/dataset/cell-data/test_pipeline/sox1_LLS_14H_1/stack/sox1_LLS_14H_1.tif"
EMBEDDING_PATH="/mnt/sda/xjh/dataset/cell-data/test_pipeline/sox1_LLS_14H_1/stack_emb"
PATTERN="*"
MODEL_TYPE="vit_l_lm"
CHECKPOINT="/mnt/sda/xjh/pt/micro_sam/models/vit_l_lm"
# for lls data
TILE_SHAPE=(1024 1024)
HALO=(256 256)

# activate environment
source activate micro-sam

micro_sam.precompute_embeddings \
    --input_path $INPUT_PATH \
    --embedding_path $EMBEDDING_PATH \
    --model_type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --tile_shape ${TILE_SHAPE[@]} \
    --halo ${HALO[@]} \
    --ndim 3 \
    --precompute_amg_state


# usage: micro_sam.precompute_embeddings [-h] -i INPUT_PATH -e EMBEDDING_PATH
#                                        [--pattern PATTERN] [-k KEY]
#                                        [-m MODEL_TYPE] [-c CHECKPOINT]
#                                        [--tile_shape TILE_SHAPE [TILE_SHAPE ...]]
#                                        [--halo HALO [HALO ...]] [-n NDIM] [-p]

# Compute the embeddings for an image.

# options:
#   -h, --help            show this help message and exit
#   -i INPUT_PATH, --input_path INPUT_PATH
#                         The filepath to the image data. Supports all data
#                         types that can be read by imageio (e.g. tif, png, ...)
#                         or elf.io.open_file (e.g. hdf5, zarr, mrc). For the
#                         latter you also need to pass the 'key' parameter.
#   -e EMBEDDING_PATH, --embedding_path EMBEDDING_PATH
#                         The path where the embeddings will be saved.
#   --pattern PATTERN     Pattern / wildcard for selecting files in a folder. To
#                         select all files use '*'.
#   -k KEY, --key KEY     The key for opening data with elf.io.open_file. This
#                         is the internal path for a hdf5 or zarr container, for
#                         an image stack it is a wild-card, e.g. '*.png' and for
#                         mrc it is 'data'.
#   -m MODEL_TYPE, --model_type MODEL_TYPE
#                         The segment anything model that will be used, one of
#                         vit_l, vit_h, vit_b, vit_t, vit_l_lm, vit_b_lm,
#                         vit_t_lm, vit_l_em_organelles, vit_b_em_organelles,
#                         vit_t_em_organelles, vit_b_histopathology,
#                         vit_l_histopathology, vit_h_histopathology,
#                         vit_b_medical_imaging, vit_l_lm_decoder,
#                         vit_b_lm_decoder, vit_t_lm_decoder,
#                         vit_l_em_organelles_decoder,
#                         vit_b_em_organelles_decoder,
#                         vit_t_em_organelles_decoder,
#                         vit_b_histopathology_decoder,
#                         vit_l_histopathology_decoder,
#                         vit_h_histopathology_decoder.
#   -c CHECKPOINT, --checkpoint CHECKPOINT
#                         Checkpoint from which the SAM model will be loaded
#                         loaded.
#   --tile_shape TILE_SHAPE [TILE_SHAPE ...]
#                         The tile shape for using tiled prediction.
#   --halo HALO [HALO ...]
#                         The halo for using tiled prediction.
#   -n NDIM, --ndim NDIM  The number of spatial dimensions in the data. Please
#                         specify this if your data has a channel dimension.
#   -p, --precompute_amg_state
#                         Whether to precompute the state for automatic instance
#                         segmentation.