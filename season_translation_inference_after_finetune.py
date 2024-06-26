import os

import torch

from swift.utils import seed_everything
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["https_proxy"] = "http://127.0.0.1:7890"

ckpt_dir = '/root/swift/output/minicpm-v-v2_5-chat/v0-20240615-142817/checkpoint-1113'
model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ["/root/autodl-tmp/val/sunny_day_bungalows_45.png"]
query0 = "Please classify objects in this image, one word for each class.(Format: [name1, name2, ..., nameN])"
response0, history0 = inference(model, template, query0, images=images)
# history0 = [[query0, response0]]
print(f'response0: {response0}')
print(f'history0: {history0}')

query1 = 'In the summer-to-winter translation task for this image, for objects in previous list, label strongly related objects as True and label irrelevant related objects as False.(Format: [(name1, Bool1), (name2, Bool2), ..., (nameN, BoolN)])'
response1, history1 = inference(model, template, query1, images=images, history=history0)
print(f'response1: {response1}')
print(f'history1: {history1}')
