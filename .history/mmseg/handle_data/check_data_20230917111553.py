import numpy as np
from PIL import Image

origin_path = '/home/lixinhao/vss/data/test/vspw/VSPW_480p/data/149_jsNS5tvjYH0/mask/00000233.png'
new_path='/home/lixinhao/vss/data/test/vspw/VSPW_480p/Detectron/train_seen/data/149_jsNS5tvjYH0/00000233.png'

labii = np.asarray(Image.open(origin_path))
new_pathii = np.asarray(Image.open(new_path))

print(np.unique(labii)!=np.unique(new_pathii))
