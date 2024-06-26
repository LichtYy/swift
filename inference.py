import os

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from swift.utils import seed_everything
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["https_proxy"] = "http://127.0.0.1:7890"


model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['datasets/val/0602.jpg']
query = 'Please identify all objects in the image and give only the result words.'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')
