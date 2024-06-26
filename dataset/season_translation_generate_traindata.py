import json
import os
import random
from PIL import Image
import argparse

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from swift.utils import seed_everything
import torch

# from season_translation_map import standardize_object_list

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["https_proxy"] = "http://127.0.0.1:7890"

model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)
# print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

standard_object_dict = {"tree": ["bush", "branch", "baretree", "bushes", "crop", "evergreentree", "evergreenbush",
                                 "forest", "foliage", "ivy", "leaf", "leaves", "plant", "shrub", "shrubbery", "tree",
                                 "vegetation"],
                        "grass": ["flower", "flowerbed", "flowerpot", "fern", "grassland", "groundcover", "grass",
                                  "lawn", "moss", "wildflower", "garden", "park"],
                        "ground": ["courtyard", "earth", "field", "ground", "land", "plaza", "soil", "sand", "terrain"],
                        "road": ["alleyway", "boardwalk", "crossing", "crosswalk", "driveway", "highway", "path",
                                 "porch", "pathway", "pavement",
                                 "road", "street", "sidewalk", "walkway"],
                        "building": ["building", "brickwork", "castle", "cabin", "church", "dwelling", "house",
                                     "logcabin", "outhouse", "shed", "tower"],
                        "stone": ["cobblestone", "menhir", "monolith", "rock", "stone", "stonehenge", "stonework"],
                        "roof": ["roofing", "roof"],
                        "vehicle": ["airplane", "bus", "boat", "bicycle", "car", "helicopter", "plane", "tram", "train",
                                    "tractor", "vehicle"],
                        "mountain": ["hillside", "hill", "mountain", "valley"],
                        "water": ["canal", "lake", "ocean", "pond", "river", "sea", "stream", "waterway", "waterfall",
                                  "water", "waterbody"],
                        "sky": ["cloud", "overcastsky", "sky"],
                        "animal": ["animal", "pedestrian", "people", "person"],
                        "stair": ["staircase", "step", "stair", "stairway"]}


def identify_obejcts(image_path):
    print(image_path, os.path.exists(image_path))
    query = 'Please classify objects in the image, one word for each class.(Format: [className1, className2, ..., classNameN])'
    # query = 'Please classify objects in the image without snow, one word for each class.(Format: [className1, className2, ..., classNameN])'
    response, history = inference(model, template, query, images=[image_path])
    print(f'response: {response}')
    response = response.split('[')[-1].split(']')[0].split(', ')
    response_1word = []
    for i in response:
        if ' ' in i:
            response_1word += i.split(' ')
        else:
            response_1word.append(i.lower())
    return list(set(response_1word))


if __name__ == '__main__':
    cfg_json = json.load(open('season_translation_cfg.json'))
    src_season = 'summer'
    tgt_season = 'winter'
    related_ol_k = cfg_json['seasons'][src_season]['related_objects_key']
    json_file_season = open(f"season_translation_traindata_{src_season}.jsonl", 'w', encoding='utf-8')
    related_ol_single = [i for related_ol_k_i in related_ol_k for i in standard_object_dict[related_ol_k_i]]
    related_ol_s = [i + 's' for i in related_ol_single]
    related_ol_es = [i + 'es' for i in related_ol_single]
    related_ol = related_ol_single + related_ol_s + related_ol_es
    count = 0
    for image_scene in ['house', 'street', 'nature']:
        for image_name in os.listdir(f"{cfg_json['dataset_root']}/{src_season}/{image_scene}"):
            ol = identify_obejcts(f"{cfg_json['dataset_root']}/{src_season}/{image_scene}/{image_name}")
            cur_data = {"query": cfg_json['season'][src_season]['query']['focus'].replace('[target]', tgt_season),
                        "response": "",
                        "images": [f"{cfg_json['dataset_root']}/{src_season}/{image_scene}/{image_name}"],
                        "history": [[cfg_json['season'][src_season]['query']['identify'], str(ol)]]}
            cur_data_response = []
            for o in ol:
                cur_data_response.append((o, o in related_ol))
            cur_data["response"] = str(cur_data_response)
            json_file_season.write(json.dumps(cur_data)+'\n')
            count += 1
            print(f"【count】:{count}")
