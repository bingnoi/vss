import mmcv
import numpy as np

origin_file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/data/0_wHveSGjXyDY/mask/00000174.png"
oc =  mmcv.imread(origin_file_name,flag="unchanged",backend="pillow")

file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/data/1385_Ycwibat6X-M/mask/00000602.png"
c =  mmcv.imread(file_name,flag="unchanged",backend="pillow")

new_file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/Detectron/train_seen/data/1385_Ycwibat6X-M/00000602.png"
cs =  mmcv.imread(new_file_name,flag="unchanged",backend="pillow")

print(np.unique(oc),np.unique(c),np.unique(cs))


v = mmcv.imread("1_result.png")
print(np.unique(v))