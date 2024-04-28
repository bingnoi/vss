import mmcv
import numpy as np

origin_file_name = "/datadisk2/lixinhao/vss/data/test/vspw/VSPW_480p/data/2225_6acPX_00M9Q/origin/00000224.jpg"
print(origin_file_name)
oc =  mmcv.imread(origin_file_name,flag="unchanged",backend="pillow")

# /datadisk2/lixinhao/vss/data/vspw/VSPW_480p/data/2307_iiBJ5rqlZ58/mask/00000054.png
file_name = "/datadisk2/lixinhao/vss/data/test/vspw/VSPW_480p/data/2225_6acPX_00M9Q/mask/00000224.png"
c =  mmcv.imread(file_name,flag="unchanged",backend="pillow")

new_file_name = "/home/lixinhao/vss/data/test/vspw/VSPW_480p/Detectron/train_seen/data/1385_Ycwibat6X-M/00000618.png"
cs =  mmcv.imread(new_file_name,flag="unchanged",backend="pillow")

print(np.unique(oc),np.unique(c),c,np.unique(cs))