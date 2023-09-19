#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import os

from multiprocessing import Pool

COCO_CATEGORIES_Seen = [
    
]

COCO_CATEGORIES_Unseen =[
{'color': [102, 8, 255], 'isthing': 1, 'id': 38, 'name': 'lake', 'trainId': 0},
{'color': [0, 122, 255], 'isthing': 1, 'id': 96, 'name': 'gun', 'trainId': 1},
{'color': [0, 224, 255], 'isthing': 1, 'id': 114, 'name': 'tissue', 'trainId': 2},
{'color': [255, 0, 112], 'isthing': 1, 'id': 112, 'name': 'tool', 'trainId': 3},
{'color': [70, 184, 160], 'isthing': 1, 'id': 116, 'name': 'computer', 'trainId': 4},
{'color': [220, 220, 220], 'isthing': 1, 'id': 28, 'name': 'other_construction', 'trainId': 5},
{'color': [255, 8, 41], 'isthing': 1, 'id': 43, 'name': 'pipeline', 'trainId': 6},
{'color': [140, 140, 140], 'isthing': 1, 'id': 49, 'name': 'car', 'trainId': 7},
{'color': [0, 61, 255], 'isthing': 1, 'id': 64, 'name': 'horse', 'trainId': 8},
{'color': [255, 0, 102], 'isthing': 1, 'id': 82, 'name': 'cage', 'trainId': 9},
{'color': [0, 102, 200], 'isthing': 1, 'id': 21, 'name': 'crosswalk', 'trainId': 10},
{'color': [0, 255, 245], 'isthing': 1, 'id': 63, 'name': 'dog', 'trainId': 11},
{'color': [235, 255, 7], 'isthing': 1, 'id': 12, 'name': 'pillar', 'trainId': 12}]

id_map = {}
for cat in COCO_CATEGORIES_Unseen:
    id_map[cat["id"]] = cat["trainId"]
    
file_train_path = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/train.txt"
file_test_path = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/test.txt"
file_val_path = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/val.txt"

train_set = set()
val_set = set()

with open(file_train_path,"r") as file:    
    for line in file:
        line = line.strip()
        train_set.add(line)
        
with open(file_test_path,"r") as file:
    for line in file:
        line = line.strip()
        train_set.add(line)
        
with open(file_val_path,"r") as file:    
    for line in file:
        line = line.strip()
        val_set.add(line)

def worker(file_tuple):
    read_path, out_path_file = file_tuple
    
    print("read: ",read_path)
    
    lab = np.asarray(Image.open(read_path))
                    
    assert lab.dtype == np.uint8

    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id in id_map:
            output[lab == obj_id] = id_map[obj_id]

    Image.fromarray(output).save(out_path_file)

if __name__ == "__main__":
    
    dataset_dir = Path() / "/home/lixinhao/vss/data/test/vspw/VSPW_480p/data" 
    
    pool = Pool(32)
    
    # for t in ["train"]:
    file_list = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            # print(root,dirs,files)
            seri = os.path.split(root)[0].split(os.path.sep)[-1]
            type_d = os.path.split(root)[1]
            # print("seri",seri,type_d)
            if(seri in val_set and type_d == 'mask'):
                # print(root,dirs,filename)
                # print(root)
                directory, s_filename = os.path.split(root)
                # print(directory.split(os.path.sep)[-2:])
                out_path = Path() / "/home/lixinhao/vss/data/test/vspw/VSPW_480p" / "Detectron" / "/".join(directory.split(os.path.sep)[-2:]) 
                out_path.mkdir(parents=True, exist_ok=True)
                out_path_file = out_path/filename
                read_path = Path() / root / filename
                
                file_list.append((read_path,out_path_file))
                
    pool.map(worker,file_list)
    print('done')
                
    # lab = np.asarray(Image.open(read_path))
    
    # assert lab.dtype == np.uint8

    # output = np.zeros_like(lab, dtype=np.uint8) + 255
    # for obj_id in np.unique(lab):
    #     if obj_id in id_map:
    #         output[lab == obj_id] = id_map[obj_id]

    # Image.fromarray(output).save(out_path_file)