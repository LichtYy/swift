export CUDA_VISIBLE_DEVICES=0
export HTTPS_PROXY=http://127.0.0.1:7890
swift sft --model_type minicpm-v-v2_5-chat --dataset /root/swift/dataset/bgi_wt_traindata_summer.jsonl --lora_target_modules ALL
