import os

import torch

from swift.utils import seed_everything
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ckpt_dir = 'output/minicpm-v-v2_5-chat/v4-20240601-145513/checkpoint-52'
model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ["datasets/val/0602.jpg"]
query = 'Give a list of areas in the picture that should be covered with snow in winter in the format [area1, area2, ..., arean]'
response, history = inference(model, template, query, images=images)
print(f'response: {response}')
print(f'history: {history}')
