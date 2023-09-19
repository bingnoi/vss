import numpy as np
from PIL import Image

origin_path = '/home/lixinhao/vss/data/test/vspw/VSPW_480p/data/127_-hIVCYO4C90/mask/00001629.png'
origin_
new_path='/home/lixinhao/vss/data/test/vspw/VSPW_480p/Detectron/val_all/data/127_-hIVCYO4C90/00001629.png'

labii = np.asarray(Image.open(origin_path))
new_pathii = np.asarray(Image.open(new_path))

print(np.unique(labii),np.unique(new_pathii))
