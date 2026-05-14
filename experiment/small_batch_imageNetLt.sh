#!/bin/bash

source ~/scripts/*.sh

RUN_DIR=/data/ltgc/augment/imagenet_lt
mkdir -p $RUN_DIR


knock "[starting] $(date) \n
python pipeline/pipeline_describe.py \
"

# run in small subtest mode to quickly verify the pipeline works, then run with full dataset
python pipeline/pipeline_describe.py \
    --d "/data/imagenet-lt/torch_image_folder/mnt/volume_sfo3_01/imagenet-lt/ImageDataset" \
    -m 10 \
    --existing_description_path $RUN_DIR/imagenet_lt_description_list.csv \
    --examples-dir $RUN_DIR/imagenet_lt_description_examples \
    --test 

knock "[starting] $(date) \n
python pipeline/pipeline_extend.py \
"

python pipeline/pipeline_extend.py \
    --existing_description_path $RUN_DIR/imagenet_lt_description_list.csv \
    --max_generate_num 10 \
    --extended_description_path $RUN_DIR/imagenet_lt_extended_description_list.csv \


knock "[starting] $(date) \n
python pipeline/pipeline_generate.py \
"

python pipeline/pipeline_generate.py \
    --extended_description_path $RUN_DIR/imagenet_lt_extended_description_list.csv \
    --data_dir $RUN_DIR/generated_imgs \
    --thresh 0.25 \
    --max_rounds 5 \
    --md $RUN_DIR/generated_example \
    --batch 7

