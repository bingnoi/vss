import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import os

from multiprocessing import Pool

file_train_path = "/home/lixinhao/vss/data/vspw/VSPW_480p/train.txt"
file_test_path = "/home/lixinhao/vss/data/vspw/VSPW_480p/test.txt"
file_val_path = "/home/lixinhao/vss/data/vspw/VSPW_480p/val.txt"

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
        
seen_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_seen.npy').tolist()
val_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_unseen.npy').tolist()
novel_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_novel.npy').tolist()

test_class = novel_cls

def worker(file_tuple):
    read_path, out_path_file = file_tuple
    
    print("read: ",read_path)
    
    lab = np.asarray(Image.open(read_path))
                    
    assert lab.dtype == np.uint8

    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id > 80:
            output[lab == obj_id] = obj_id

    Image.fromarray(output).save(out_path_file)

if __name__ == "__main__":
    
    dataset_dir = Path() / "/home/lixinhao/vss/data/vspw/VSPW_480p/data" 
    
    pool = Pool(32)
    
    file_list = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            seri = os.path.split(root)[0].split(os.path.sep)[-1]
            type_d = os.path.split(root)[1]
            if(seri in val_set and type_d == 'mask'):
                directory, s_filename = os.path.split(root)
                out_path = Path() / "/home/lixinhao/vss/data/vspw/VSPW_480p" / "Detectron" / "val_unseen" /"/".join(directory.split(os.path.sep)[-2:]) 
                out_path.mkdir(parents=True, exist_ok=True)
                out_path_file = out_path/filename
                read_path = Path() / root / filename
                
                file_list.append((read_path,out_path_file))
                
    pool.map(worker,file_list)
    print('done')