import mmcv
import numpy as np

origin_file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/data/127_-hIVCYO4C90/mask/00000587.png"
oc =  mmcv.imread(origin_file_name,flag="unchanged",backend="pillow")

file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/data/127_-hIVCYO4C90/mask/00000618.png"
c =  mmcv.imread(file_name,flag="unchanged",backend="pillow")

new_file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/Detectron/train_seen/data/1385_Ycwibat6X-M/00000618.png"
cs =  mmcv.imread(new_file_name,flag="unchanged",backend="pillow")

print(np.unique(oc),np.unique(c),np.unique(cs))