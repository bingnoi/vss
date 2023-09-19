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

COCO_CATEGORIES_Seen = 
id_map = {}
for cat in COCO_CATEGORIES_Seen:
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
        
with open(file_val_path,"r") as file:    
    for line in file:
        line = line.strip()
        val_set.add(line)

def worker(file_tuple):
    file, output_file = file_tuple
    lab = np.asarray(Image.open(file))
    assert lab.dtype == np.uint8

    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id in id_map:
            output[lab == obj_id] = id_map[obj_id]

    Image.fromarray(output).save(output_file)

if __name__ == "__main__":
    
    dataset_dir = Path() / "/home/lixinhao/vss/data/test/vspw/VSPW_480p/data" 
    
    pool = Pool(32)
    
    for t in ["train"]:
        file_list = []
        for root, dirs, files in os.walk(dataset_dir):
            for filename in files:
                # print(root,dirs,files)
                seri = os.path.split(root)[0].split(os.path.sep)[-1]
                type_d = os.path.split(root)[1]
                # print("seri",seri,type_d)
                if(seri in train_set and type_d == 'mask'):
                    # print(root,dirs,filename)
                    # print(root)
                    directory, s_filename = os.path.split(root)
                    # print(directory.split(os.path.sep)[-2:])
                    out_path = Path() / "/home/lixinhao/vss/data/test/vspw/VSPW_480p" / "Detectron" / "/".join(directory.split(os.path.sep)[-2:]) 
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path_file = out_path/filename
                    read_path = Path() / root / filename
                    
                    print(read_path)
                    # exit()
                    
                    lab = np.asarray(Image.open(read_path))
                    
                    assert lab.dtype == np.uint8

                    output = np.zeros_like(lab, dtype=np.uint8) + 255
                    for obj_id in np.unique(lab):
                        if obj_id in id_map:
                            output[lab == obj_id] = id_map[obj_id]

                    Image.fromarray(output).save(out_path_file)