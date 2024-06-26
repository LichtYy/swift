export CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:7890
swift infer --ckpt_dir output/minicpm-v-v2_5-chat/v1-20240527-114145/checkpoint-309 --load_dataset_config true