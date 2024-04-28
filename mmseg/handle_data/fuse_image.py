import cv2
import numpy as np
from queue import Queue
import random
from PIL import Image

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]
               ]

palette_list = []
for kk in PALETTE:
    palette_list=palette_list+kk

def bfs_expand(image, start_point, num_points, surrounding_colors):
    height, width = image.shape
    # image = np.array(image)
    visited = np.zeros((height, width), dtype=bool)
    queue = [(start_point[0], start_point[1])]
    
    region = []

    visited[start_point[0], start_point[1]] = True
    region.append(start_point)

    while len(region) < num_points and queue:
        current_point = queue.pop(0)

        # 随机打乱四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)

        for direction in directions:
            next_point = (current_point[0] + direction[0], current_point[1] + direction[1])

            # 检查下一个点是否在图像范围内且未被访问过
            if 0 <= next_point[0] < height and 0 <= next_point[1] < width and not visited[next_point[0], next_point[1]]:
                queue.append(next_point)
                visited[next_point[0], next_point[1]] = True
                region.append(next_point)

    # 将扩散得到的所有点的颜色设置为画面上其他点的颜色
    for point in region:
        image[point[0], point[1]] = random.choice(surrounding_colors)

    return region

def collect_n_regions(image, num_regions, points_per_region):
    height, width = image.shape
    all_points = []

    for _ in range(num_regions):
        # 随机选取一个起始点（从图像边界选择）
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            start_point = (random.choice([0, height-1]), random.randint(0, width-1))
        else:
            start_point = (random.randint(0, height-1), random.choice([0, width-1]))

        # 获取周围点的颜色
        surrounding_colors = get_surrounding_colors(image, start_point)

        # 扩散获取指定数量的点，并将颜色设置为周围点的颜色
        region = bfs_expand(image, start_point, points_per_region, surrounding_colors)

        # 将当前区域的所有点添加到结果数组中
        all_points.extend(region)

    return all_points

def get_surrounding_colors(image, center_point):
    # print(image.size)
    height, width = image.shape
    # image = np.array(image)
    # print(image.shape)
    surrounding_points = []

    # 定义上、下、左、右四个方向的偏移量
    directions = [(-1,1),(1,1),(-1, 0), (1, 0),(1,-1),(-1,-1) ,(0, -1), (0, 1)]

    for direction in directions:
        next_point = (center_point[0] + direction[0], center_point[1] + direction[1])

        # 检查下一个点是否在图像范围内
        if 0 <= next_point[0] < height and 0 <= next_point[1] < width:
            surrounding_points.append(image[next_point[0], next_point[1]])

    return surrounding_points

def merge_images(original_image_path, segmentation_gt_path, output_path, alpha=0.3, colormap=cv2.COLORMAP_JET): 
    # print(original_image_path,segmentation_gt_path)
    print(output_path)
    
    # 读取原始图像和分割的ground truth 
    # original_image = cv2.imread(original_image_path) 
    # segmentation_gt = cv2.imread(segmentation_gt_path, cv2.IMREAD_GRAYSCALE) 

    img=Image.open(segmentation_gt_path)
    img=np.array(img)

    # res = Image.fromarray(img.astype(np.uint8), mode='P')
    # res.putpalette(palette_list)
    segmentation_gt_colored = img

    np.random.seed(42)

    noisy_mask = segmentation_gt_colored.copy()

    noisy_mask = np.array(noisy_mask)
    all_point = collect_n_regions(noisy_mask, 20, 13999)

    noisy_mask = Image.fromarray(noisy_mask.astype(np.uint8),mode='P')
    
    noisy_mask.putpalette(palette_list)
    noisy_mask.save(output_path)


# 示例调用，使用 COLORMAP_JET 作为颜色映射表

# /datadisk2/lixinhao/vss/data/vspw/VSPW_480p/data/29_I_zH7gzZ0WI

root = "eccv_img_new"
save_name = "video_img"

import os
# for y in sorted(os.listdir(f"/datadisk2/lixinhao/vss/{root}")):
#     for i in os.listdir(os.path.join(f"/datadisk2/lixinhao/vss/{root}",y)+"/origin/")[:20]:
#         i = i.split(".")[0]
#         # print(f'/datadisk2/lixinhao/vss/data/vspw/VSPW_480p/data/29_I_zH7gzZ0WI/origin/{i}.jpg')
#         save_dir = f'/datadisk2/lixinhao/vss/{root}/{y}/{save_name}'
#         # os.makedirs(directory_path, exist_ok=True)
#         os.makedirs(save_dir,exist_ok=True)
#         merge_images(f'/datadisk2/lixinhao/vss/{root}/{y}/origin/{i}.jpg', f'/datadisk2/lixinhao/vss/{root}/{y}/mask/{i}.png', f'{save_dir}/{i}.png', colormap=cv2.COLORMAP_JET)

for i in sorted(os.listdir(f"/datadisk2/lixinhao/vss/{root}/origin"))[:20]:
    i = i.split(".")[0]
    # print(f'/datadisk2/lixinhao/vss/data/vspw/VSPW_480p/data/29_I_zH7gzZ0WI/origin/{i}.jpg')
    merge_images(f'/datadisk2/lixinhao/vss/{root}/origin/{i}.jpg', f'/datadisk2/lixinhao/vss/{root}/mask/{i}.png', f'/datadisk2/lixinhao/vss/{root}/{save_name}/{i}.png', colormap=cv2.COLORMAP_JET)
