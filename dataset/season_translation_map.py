# 仅不记录纯s复数结尾，键为代表单词，类内各单词的分割图共用
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


def standardize_object_list(old_list, list_result=False, standard_dict=None, record_unclassified=False):
    if standard_dict is None:
        standard_dict = standard_object_dict
    new_dict = {k: [] for k in standard_dict.keys()}
    if record_unclassified:
        new_dict['-1'] = []  # 未分类的其他物体
    # old_list = set()
    # old_list_str = old_list_str.strip('[]\n').split(', ')
    # for i in old_list_str:
    #     old_list.add(i.lower())
    for o in list(old_list):
        flag = False
        for k, v in standard_dict.items():
            for i in v:
                if o == i:
                    new_dict[k].append(o)
                    flag = True
                    break
                elif o[-1] == 's' and o[:-1] == i:
                    new_dict[k].append(o)
                    flag = True
                    break
        if record_unclassified and not flag:
            new_dict['-1'].append(o)
    if list_result:
        return [k for k in new_dict.keys() if len(new_dict[k]) > 0]
    return new_dict
