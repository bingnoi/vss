#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import os

from shutil import copyfile


COCO_CATEGORIES_Seen = [
    
]

COCO_CATEGORIES_Unseen = [{
    'color': [235, 255, 7],
    'isthing': 1,
    'id': 12,
    'name': 'pillar',
    'trainId': 0
}, {
    'color': [0, 102, 200],
    'isthing': 1,
    'id': 21,
    'name': 'crosswalk',
    'trainId': 1
}, {
    'color': [220, 220, 220],
    'isthing': 1,
    'id': 28,
    'name': 'other_construction',
    'trainId': 2
}, {
    'color': [102, 8, 255],
    'isthing': 1,
    'id': 38,
    'name': 'lake',
    'trainId': 3
}, {
    'color': [255, 8, 41],
    'isthing': 1,
    'id': 43,
    'name': 'pipeline',
    'trainId': 4
}, {
    'color': [140, 140, 140],
    'isthing': 1,
    'id': 49,
    'name': 'car',
    'trainId': 5
}, {
    'color': [0, 255, 245],
    'isthing': 1,
    'id': 63,
    'name': 'dog',
    'trainId': 6
}, {
    'color': [0, 61, 255],
    'isthing': 1,
    'id': 64,
    'name': 'horse',
    'trainId': 7
}, {
    'color': [255, 0, 102],
    'isthing': 1,
    'id': 82,
    'name': 'cage',
    'trainId': 8
}, {
    'color': [0, 122, 255],
    'isthing': 1,
    'id': 96,
    'name': 'gun',
    'trainId': 9
}, {
    'color': [255, 0, 112],
    'isthing': 1,
    'id': 112,
    'name': 'tool',
    'trainId': 10
}, {
    'color': [0, 224, 255],
    'isthing': 1,
    'id': 114,
    'name': 'tissue',
    'trainId': 11
}, {
    'color': [70, 184, 160],
    'isthing': 1,
    'id': 116,
    'name': 'computer',
    'trainId': 12
}]

for item in COCO_CATEGORIES_Unseen:
    item['trainId'] = item['trainId'] + 156

COCO_CATEGORIES_ALL = COCO_CATEGORIES_Seen + COCO_CATEGORIES_Unseen

if __name__ == "__main__":
    dataset_dir = "/datadisk/lixinhao" / Path(os.getenv("DETECTRON2_DATASETS", "dataset")) / "coco" / "coco_stuff"

    id_map = {}
    for cat in COCO_CATEGORIES_ALL:
        id_map[cat["id"]] = cat["trainId"]

    name = "val2017_all"
    annotation_dir = dataset_dir / "annotations" / "val2017"
    output_dir = dataset_dir / "annotations_detectron2" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm.tqdm(list(annotation_dir.iterdir())):
        output_file = output_dir / file.name
        lab = np.asarray(Image.open(file))
        assert lab.dtype == np.uint8

        output = np.zeros_like(lab, dtype=np.uint8) + 255
        for obj_id in np.unique(lab):
            if obj_id in id_map:
                output[lab == obj_id] = id_map[obj_id]

        Image.fromarray(output).save(output_file)