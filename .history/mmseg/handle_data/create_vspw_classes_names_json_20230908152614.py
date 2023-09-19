import os
import numpy as np
import json
import copy

VSPW_CATEGORIES = 
for item in VSPW_CATEGORIES:
    item['id'] = item['id'] - 1

COCO_CATEGORIES_Seen = []
COCO_CATEGORIES_unseen = []

seen_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_seen.npy')
val_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_unseen.npy')
novel_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_novel.npy')


train_cls = seen_cls.tolist() + val_cls.tolist()

seen_classnames = []
unseen_classnames = []

count_train = 0
count_test = 0
for item in VSPW_CATEGORIES:
    id = item['id']
    name = item['name']

    if id in train_cls:
        seen_classnames.append(name)
        tmp = copy.deepcopy(item)
        tmp['trainId'] = count_train
        COCO_CATEGORIES_Seen.append(tmp)
        count_train = count_train + 1
    else:
        unseen_classnames.append(name)
        tmp = copy.deepcopy(item)
        tmp['trainId'] = count_test
        COCO_CATEGORIES_unseen.append(tmp)
        count_test = count_test + 1

with open(r'/home/lixinhao/vss/mmseg/handle_data/seen_classnames.json', 'w') as f_out:
    json.dump(seen_classnames, f_out)

with open(r'/home/lixinhao/vss/mmseg/handle_data/unseen_classnames.json', 'w') as f_out:
    json.dump(unseen_classnames, f_out)

with open(r'/home/lixinhao/vss/mmseg/handle_data/all_classnames.json', 'w') as f_out:
    json.dump(seen_classnames + unseen_classnames, f_out)

# print(COCO_CATEGORIES_Seen)

# print(COCO_CATEGORIES_unseen)

