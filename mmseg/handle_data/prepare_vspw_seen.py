#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
import os

from multiprocessing import Pool,Manager

# COCO_CATEGORIES_Seen = [{
#     'color': [120, 120, 120],
#     'isthing': 1,
#     'id': 1,
#     'name': 'wall',
#     'trainId': 0
# }, {
#     'color': [180, 120, 120],
#     'isthing': 1,
#     'id': 2,
#     'name': 'ceiling',
#     'trainId': 1
# }, {
#     'color': [6, 230, 230],
#     'isthing': 1,
#     'id': 3,
#     'name': 'door',
#     'trainId': 2
# }, {
#     'color': [80, 50, 50],
#     'isthing': 1,
#     'id': 4,
#     'name': 'stair',
#     'trainId': 3
# }, {
#     'color': [4, 200, 3],
#     'isthing': 1,
#     'id': 5,
#     'name': 'ladder',
#     'trainId': 4
# }, {
#     'color': [120, 120, 80],
#     'isthing': 1,
#     'id': 6,
#     'name': 'escalator',
#     'trainId': 5
# }, {
#     'color': [140, 140, 140],
#     'isthing': 1,
#     'id': 7,
#     'name': 'Playground_slide',
#     'trainId': 6
# }, {
#     'color': [204, 5, 255],
#     'isthing': 1,
#     'id': 8,
#     'name': 'handrail_or_fence',
#     'trainId': 7
# }, {
#     'color': [230, 230, 230],
#     'isthing': 1,
#     'id': 9,
#     'name': 'window',
#     'trainId': 8
# }, {
#     'color': [4, 250, 7],
#     'isthing': 1,
#     'id': 10,
#     'name': 'rail',
#     'trainId': 9
# }, {
#     'color': [224, 5, 255],
#     'isthing': 1,
#     'id': 11,
#     'name': 'goal',
#     'trainId': 10
# }, {
#     'color': [150, 5, 61],
#     'isthing': 1,
#     'id': 13,
#     'name': 'pole',
#     'trainId': 11
# }, {
#     'color': [120, 120, 70],
#     'isthing': 1,
#     'id': 14,
#     'name': 'floor',
#     'trainId': 12
# }, {
#     'color': [8, 255, 51],
#     'isthing': 1,
#     'id': 15,
#     'name': 'ground',
#     'trainId': 13
# }, {
#     'color': [255, 6, 82],
#     'isthing': 1,
#     'id': 16,
#     'name': 'grass',
#     'trainId': 14
# }, {
#     'color': [143, 255, 140],
#     'isthing': 1,
#     'id': 17,
#     'name': 'sand',
#     'trainId': 15
# }, {
#     'color': [204, 255, 4],
#     'isthing': 1,
#     'id': 18,
#     'name': 'athletic_field',
#     'trainId': 16
# }, {
#     'color': [255, 51, 7],
#     'isthing': 1,
#     'id': 19,
#     'name': 'road',
#     'trainId': 17
# }, {
#     'color': [204, 70, 3],
#     'isthing': 1,
#     'id': 20,
#     'name': 'path',
#     'trainId': 18
# }, {
#     'color': [61, 230, 250],
#     'isthing': 1,
#     'id': 22,
#     'name': 'building',
#     'trainId': 19
# }, {
#     'color': [255, 6, 51],
#     'isthing': 1,
#     'id': 23,
#     'name': 'house',
#     'trainId': 20
# }, {
#     'color': [11, 102, 255],
#     'isthing': 1,
#     'id': 24,
#     'name': 'bridge',
#     'trainId': 21
# }, {
#     'color': [255, 7, 71],
#     'isthing': 1,
#     'id': 25,
#     'name': 'tower',
#     'trainId': 22
# }, {
#     'color': [255, 9, 224],
#     'isthing': 1,
#     'id': 26,
#     'name': 'windmill',
#     'trainId': 23
# }, {
#     'color': [9, 7, 230],
#     'isthing': 1,
#     'id': 27,
#     'name': 'well_or_well_lid',
#     'trainId': 24
# }, {
#     'color': [255, 9, 92],
#     'isthing': 1,
#     'id': 29,
#     'name': 'sky',
#     'trainId': 25
# }, {
#     'color': [112, 9, 255],
#     'isthing': 1,
#     'id': 30,
#     'name': 'mountain',
#     'trainId': 26
# }, {
#     'color': [8, 255, 214],
#     'isthing': 1,
#     'id': 31,
#     'name': 'stone',
#     'trainId': 27
# }, {
#     'color': [7, 255, 224],
#     'isthing': 1,
#     'id': 32,
#     'name': 'wood',
#     'trainId': 28
# }, {
#     'color': [255, 184, 6],
#     'isthing': 1,
#     'id': 33,
#     'name': 'ice',
#     'trainId': 29
# }, {
#     'color': [10, 255, 71],
#     'isthing': 1,
#     'id': 34,
#     'name': 'snowfield',
#     'trainId': 30
# }, {
#     'color': [255, 41, 10],
#     'isthing': 1,
#     'id': 35,
#     'name': 'grandstand',
#     'trainId': 31
# }, {
#     'color': [7, 255, 255],
#     'isthing': 1,
#     'id': 36,
#     'name': 'sea',
#     'trainId': 32
# }, {
#     'color': [224, 255, 8],
#     'isthing': 1,
#     'id': 37,
#     'name': 'river',
#     'trainId': 33
# }, {
#     'color': [255, 61, 6],
#     'isthing': 1,
#     'id': 39,
#     'name': 'waterfall',
#     'trainId': 34
# }, {
#     'color': [255, 194, 7],
#     'isthing': 1,
#     'id': 40,
#     'name': 'water',
#     'trainId': 35
# }, {
#     'color': [255, 122, 8],
#     'isthing': 1,
#     'id': 41,
#     'name': 'billboard_or_Bulletin_Board',
#     'trainId': 36
# }, {
#     'color': [0, 255, 20],
#     'isthing': 1,
#     'id': 42,
#     'name': 'sculpture',
#     'trainId': 37
# }, {
#     'color': [255, 5, 153],
#     'isthing': 1,
#     'id': 44,
#     'name': 'flag',
#     'trainId': 38
# }, {
#     'color': [6, 51, 255],
#     'isthing': 1,
#     'id': 45,
#     'name': 'parasol_or_umbrella',
#     'trainId': 39
# }, {
#     'color': [235, 12, 255],
#     'isthing': 1,
#     'id': 46,
#     'name': 'cushion_or_carpet',
#     'trainId': 40
# }, {
#     'color': [160, 150, 20],
#     'isthing': 1,
#     'id': 47,
#     'name': 'tent',
#     'trainId': 41
# }, {
#     'color': [0, 163, 255],
#     'isthing': 1,
#     'id': 48,
#     'name': 'roadblock',
#     'trainId': 42
# }, {
#     'color': [250, 10, 15],
#     'isthing': 1,
#     'id': 50,
#     'name': 'bus',
#     'trainId': 43
# }, {
#     'color': [20, 255, 0],
#     'isthing': 1,
#     'id': 51,
#     'name': 'truck',
#     'trainId': 44
# }, {
#     'color': [31, 255, 0],
#     'isthing': 1,
#     'id': 52,
#     'name': 'bicycle',
#     'trainId': 45
# }, {
#     'color': [255, 31, 0],
#     'isthing': 1,
#     'id': 53,
#     'name': 'motorcycle',
#     'trainId': 46
# }, {
#     'color': [255, 224, 0],
#     'isthing': 1,
#     'id': 54,
#     'name': 'wheeled_machine',
#     'trainId': 47
# }, {
#     'color': [153, 255, 0],
#     'isthing': 1,
#     'id': 55,
#     'name': 'ship_or_boat',
#     'trainId': 48
# }, {
#     'color': [0, 0, 255],
#     'isthing': 1,
#     'id': 56,
#     'name': 'raft',
#     'trainId': 49
# }, {
#     'color': [255, 71, 0],
#     'isthing': 1,
#     'id': 57,
#     'name': 'airplane',
#     'trainId': 50
# }, {
#     'color': [0, 235, 255],
#     'isthing': 1,
#     'id': 58,
#     'name': 'tyre',
#     'trainId': 51
# }, {
#     'color': [0, 173, 255],
#     'isthing': 1,
#     'id': 59,
#     'name': 'traffic_light',
#     'trainId': 52
# }, {
#     'color': [31, 0, 255],
#     'isthing': 1,
#     'id': 60,
#     'name': 'lamp',
#     'trainId': 53
# }, {
#     'color': [11, 200, 200],
#     'isthing': 1,
#     'id': 61,
#     'name': 'person',
#     'trainId': 54
# }, {
#     'color': [255, 82, 0],
#     'isthing': 1,
#     'id': 62,
#     'name': 'cat',
#     'trainId': 55
# }, {
#     'color': [0, 255, 112],
#     'isthing': 1,
#     'id': 65,
#     'name': 'cattle',
#     'trainId': 56
# }, {
#     'color': [0, 255, 133],
#     'isthing': 1,
#     'id': 66,
#     'name': 'other_animal',
#     'trainId': 57
# }, {
#     'color': [255, 0, 0],
#     'isthing': 1,
#     'id': 67,
#     'name': 'tree',
#     'trainId': 58
# }, {
#     'color': [255, 163, 0],
#     'isthing': 1,
#     'id': 68,
#     'name': 'flower',
#     'trainId': 59
# }, {
#     'color': [255, 102, 0],
#     'isthing': 1,
#     'id': 69,
#     'name': 'other_plant',
#     'trainId': 60
# }, {
#     'color': [194, 255, 0],
#     'isthing': 1,
#     'id': 70,
#     'name': 'toy',
#     'trainId': 61
# }, {
#     'color': [0, 143, 255],
#     'isthing': 1,
#     'id': 71,
#     'name': 'ball_net',
#     'trainId': 62
# }, {
#     'color': [51, 255, 0],
#     'isthing': 1,
#     'id': 72,
#     'name': 'backboard',
#     'trainId': 63
# }, {
#     'color': [0, 82, 255],
#     'isthing': 1,
#     'id': 73,
#     'name': 'skateboard',
#     'trainId': 64
# }, {
#     'color': [0, 255, 41],
#     'isthing': 1,
#     'id': 74,
#     'name': 'bat',
#     'trainId': 65
# }, {
#     'color': [0, 255, 173],
#     'isthing': 1,
#     'id': 75,
#     'name': 'ball',
#     'trainId': 66
# }, {
#     'color': [10, 0, 255],
#     'isthing': 1,
#     'id': 76,
#     'name': 'cupboard_or_showcase_or_storage_rack',
#     'trainId': 67
# }, {
#     'color': [173, 255, 0],
#     'isthing': 1,
#     'id': 77,
#     'name': 'box',
#     'trainId': 68
# }, {
#     'color': [0, 255, 153],
#     'isthing': 1,
#     'id': 78,
#     'name': 'traveling_case_or_trolley_case',
#     'trainId': 69
# }, {
#     'color': [255, 92, 0],
#     'isthing': 1,
#     'id': 79,
#     'name': 'basket',
#     'trainId': 70
# }, {
#     'color': [255, 0, 255],
#     'isthing': 1,
#     'id': 80,
#     'name': 'bag_or_package',
#     'trainId': 71
# }, {
#     'color': [255, 0, 245],
#     'isthing': 1,
#     'id': 81,
#     'name': 'trash_can',
#     'trainId': 72
# }, {
#     'color': [255, 173, 0],
#     'isthing': 1,
#     'id': 83,
#     'name': 'plate',
#     'trainId': 73
# }, {
#     'color': [255, 0, 20],
#     'isthing': 1,
#     'id': 84,
#     'name': 'tub_or_bowl_or_pot',
#     'trainId': 74
# }, {
#     'color': [255, 184, 184],
#     'isthing': 1,
#     'id': 85,
#     'name': 'bottle_or_cup',
#     'trainId': 75
# }, {
#     'color': [0, 31, 255],
#     'isthing': 1,
#     'id': 86,
#     'name': 'barrel',
#     'trainId': 76
# }, {
#     'color': [0, 255, 61],
#     'isthing': 1,
#     'id': 87,
#     'name': 'fishbowl',
#     'trainId': 77
# }, {
#     'color': [0, 71, 255],
#     'isthing': 1,
#     'id': 88,
#     'name': 'bed',
#     'trainId': 78
# }, {
#     'color': [255, 0, 204],
#     'isthing': 1,
#     'id': 89,
#     'name': 'pillow',
#     'trainId': 79
# }, {
#     'color': [0, 255, 194],
#     'isthing': 1,
#     'id': 90,
#     'name': 'table_or_desk',
#     'trainId': 80
# }, {
#     'color': [0, 255, 82],
#     'isthing': 1,
#     'id': 91,
#     'name': 'chair_or_seat',
#     'trainId': 81
# }, {
#     'color': [0, 10, 255],
#     'isthing': 1,
#     'id': 92,
#     'name': 'bench',
#     'trainId': 82
# }, {
#     'color': [0, 112, 255],
#     'isthing': 1,
#     'id': 93,
#     'name': 'sofa',
#     'trainId': 83
# }, {
#     'color': [51, 0, 255],
#     'isthing': 1,
#     'id': 94,
#     'name': 'shelf',
#     'trainId': 84
# }, {
#     'color': [0, 194, 255],
#     'isthing': 1,
#     'id': 95,
#     'name': 'bathtub',
#     'trainId': 85
# }, {
#     'color': [0, 255, 163],
#     'isthing': 1,
#     'id': 97,
#     'name': 'commode',
#     'trainId': 86
# }, {
#     'color': [255, 153, 0],
#     'isthing': 1,
#     'id': 98,
#     'name': 'roaster',
#     'trainId': 87
# }, {
#     'color': [0, 255, 10],
#     'isthing': 1,
#     'id': 99,
#     'name': 'other_machine',
#     'trainId': 88
# }, {
#     'color': [255, 112, 0],
#     'isthing': 1,
#     'id': 100,
#     'name': 'refrigerator',
#     'trainId': 89
# }, {
#     'color': [143, 255, 0],
#     'isthing': 1,
#     'id': 101,
#     'name': 'washing_machine',
#     'trainId': 90
# }, {
#     'color': [82, 0, 255],
#     'isthing': 1,
#     'id': 102,
#     'name': 'Microwave_oven',
#     'trainId': 91
# }, {
#     'color': [163, 255, 0],
#     'isthing': 1,
#     'id': 103,
#     'name': 'fan',
#     'trainId': 92
# }, {
#     'color': [255, 235, 0],
#     'isthing': 1,
#     'id': 104,
#     'name': 'curtain',
#     'trainId': 93
# }, {
#     'color': [8, 184, 170],
#     'isthing': 1,
#     'id': 105,
#     'name': 'textiles',
#     'trainId': 94
# }, {
#     'color': [133, 0, 255],
#     'isthing': 1,
#     'id': 106,
#     'name': 'clothes',
#     'trainId': 95
# }, {
#     'color': [0, 255, 92],
#     'isthing': 1,
#     'id': 107,
#     'name': 'painting_or_poster',
#     'trainId': 96
# }, {
#     'color': [184, 0, 255],
#     'isthing': 1,
#     'id': 108,
#     'name': 'mirror',
#     'trainId': 97
# }, {
#     'color': [255, 0, 31],
#     'isthing': 1,
#     'id': 109,
#     'name': 'flower_pot_or_vase',
#     'trainId': 98
# }, {
#     'color': [0, 184, 255],
#     'isthing': 1,
#     'id': 110,
#     'name': 'clock',
#     'trainId': 99
# }, {
#     'color': [0, 214, 255],
#     'isthing': 1,
#     'id': 111,
#     'name': 'book',
#     'trainId': 100
# }, {
#     'color': [92, 255, 0],
#     'isthing': 1,
#     'id': 113,
#     'name': 'blackboard',
#     'trainId': 101
# }, {
#     'color': [112, 224, 255],
#     'isthing': 1,
#     'id': 115,
#     'name': 'screen_or_television',
#     'trainId': 102
# }, {
#     'color': [163, 0, 255],
#     'isthing': 1,
#     'id': 117,
#     'name': 'printer',
#     'trainId': 103
# }, {
#     'color': [153, 0, 255],
#     'isthing': 1,
#     'id': 118,
#     'name': 'Mobile_phone',
#     'trainId': 104
# }, {
#     'color': [71, 255, 0],
#     'isthing': 1,
#     'id': 119,
#     'name': 'keyboard',
#     'trainId': 105
# }, {
#     'color': [255, 0, 163],
#     'isthing': 1,
#     'id': 120,
#     'name': 'other_electronic_product',
#     'trainId': 106
# }, {
#     'color': [255, 204, 0],
#     'isthing': 1,
#     'id': 121,
#     'name': 'fruit',
#     'trainId': 107
# }, {
#     'color': [255, 0, 143],
#     'isthing': 1,
#     'id': 122,
#     'name': 'food',
#     'trainId': 108
# }, {
#     'color': [0, 255, 235],
#     'isthing': 1,
#     'id': 123,
#     'name': 'instrument',
#     'trainId': 109
# }, {
#     'color': [133, 255, 0],
#     'isthing': 1,
#     'id': 124,
#     'name': 'train',
#     'trainId': 110
# }]
# COCO_CATEGORIES_Seen = {"others": "0", "wall": "1", "ceiling": "2", "door": "3", "stair": "4", "ladder": "5", 
#     "escalator": "6", "Playground_slide": "7", "handrail_or_fence": "8", "window": "9", 
#     "rail": "10", "goal": "11", "pillar": "12", "pole": "13", "floor": "14",
#     "ground": "15", "grass": "16", "sand": "17", "athletic_field": "18", "road": "19", "path": "20",
#     "crosswalk": "21", "building": "22", "house": "23", "bridge": "24", "tower": "25", "windmill": "26",
#     "well_or_well_lid": "27", "other_construction": "28", "sky": "29", "mountain": "30", "stone": "31",
#     "wood": "32", "ice": "33", "snowfield": "34", "grandstand": "35", "sea": "36", "river": "37", 
#     "lake": "38", "waterfall": "39", "water": "40", "billboard_or_Bulletin_Board": "41", "sculpture": "42",
#     "pipeline": "43", "flag": "44", "parasol_or_umbrella": "45", "cushion_or_carpet": "46", "tent": "47",
#     "roadblock": "48", "car": "49", "bus": "50", "truck": "51", "bicycle": "52", "motorcycle": "53",
#     "wheeled_machine": "54", "ship_or_boat": "55", "raft": "56", "airplane": "57", "tyre": "58",
#     "traffic_light": "59", "lamp": "60", "person": "61", "cat": "62", "dog": "63", "horse": "64",
#     "cattle": "65", "other_animal": "66", "tree": "67", "flower": "68", "other_plant": "69", "toy": "70",
#     "ball_net": "71", "backboard": "72", "skateboard": "73", "bat": "74", "ball": "75",
#     "cupboard_or_showcase_or_storage_rack": "76", "box": "77", "traveling_case_or_trolley_case": "78",
#     "basket": "79", "bag_or_package": "80", "trash_can": "81", "cage": "82", "plate": "83",
#     "tub_or_bowl_or_pot": "84", "bottle_or_cup": "85", "barrel": "86", "fishbowl": "87", "bed": "88",
#     "pillow": "89", "table_or_desk": "90", "chair_or_seat": "91", "bench": "92", "sofa": "93",
#     "shelf": "94", "bathtub": "95", "gun": "96", "commode": "97", "roaster": "98", "other_machine": "99",
#     "refrigerator": "100", "washing_machine": "101", "Microwave_oven": "102", "fan": "103", "curtain": "104",
#     "textiles": "105", "clothes": "106", "painting_or_poster": "107", "mirror": "108", "flower_pot_or_vase": "109",
#     "clock": "110", "book": "111", "tool": "112", "blackboard": "113", "tissue": "114", "screen_or_television": "115",
#     "computer": "116", "printer": "117", "Mobile_phone": "118", "keyboard": "119", "other_electronic_product": "120",
#     "fruit": "121", "food": "122", "instrument": "123", "train": "124"}

COCO_CATEGORIES_Seen = [{'color': [120, 120, 120], 'id': 1, 'name': 'wall', 'trainid': 1}, {'color': [180, 120, 120], 'id': 2, 'name': 'ceiling', 'trainid': 2}, {'color': [6, 230, 230], 'id': 3, 'name': 'door', 'trainid': 3}, {'color': [80, 50, 50], 'id': 4, 'name': 'stair', 'trainid': 4}, {'color': [4, 200, 3], 'id': 5, 'name': 'ladder', 'trainid': 5}, {'color': [120, 120, 80], 'id': 6, 'name': 'escalator', 'trainid': 6}, {'color': [140, 140, 140], 'id': 7, 'name': 'Playground_slide', 'trainid': 7}, {'color': [204, 5, 255], 'id': 8, 'name': 'handrail_or_fence', 'trainid': 8}, {'color': [230, 230, 230], 'id': 9, 'name': 'window', 'trainid': 9}, {'color': [4, 250, 7], 'id': 10, 'name': 'rail', 'trainid': 10}, {'color': [224, 5, 255], 'id': 11, 'name': 'goal', 'trainid': 11}, {'color': [150, 5, 61], 'id': 13, 'name': 'pole', 'trainid': 12}, {'color': [120, 120, 70], 'id': 14, 'name': 'floor', 'trainid': 13}, {'color': [8, 255, 51], 'id': 15, 'name': 'ground', 'trainid': 14}, {'color': [255, 6, 82], 'id': 16, 'name': 'grass', 'trainid': 15}, {'color': [143, 255, 140], 'id': 17, 'name': 'sand', 'trainid': 16}, {'color': [204, 255, 4], 'id': 18, 'name': 'athletic_field', 'trainid': 17}, {'color': [255, 51, 7], 'id': 19, 'name': 'road', 'trainid': 18}, {'color': [204, 70, 3], 'id': 20, 'name': 'path', 'trainid': 19}, {'color': [61, 230, 250], 'id': 22, 'name': 'building', 'trainid': 20}, {'color': [255, 6, 51], 'id': 23, 'name': 'house', 'trainid': 21}, {'color': [11, 102, 255], 'id': 24, 'name': 'bridge', 'trainid': 22}, {'color': [255, 7, 71], 'id': 25, 'name': 'tower', 'trainid': 23}, {'color': [255, 9, 224], 'id': 26, 'name': 'windmill', 'trainid': 24}, {'color': [9, 7, 230], 'id': 27, 'name': 'well_or_well_lid', 'trainid': 25}, {'color': [255, 9, 92], 'id': 29, 'name': 'sky', 'trainid': 26}, {'color': [112, 9, 255], 'id': 30, 'name': 'mountain', 'trainid': 27}, {'color': [8, 255, 214], 'id': 31, 'name': 'stone', 'trainid': 28}, {'color': [7, 255, 224], 'id': 32, 'name': 'wood', 'trainid': 29}, {'color': [255, 184, 6], 'id': 33, 'name': 'ice', 'trainid': 30}, {'color': [10, 255, 71], 'id': 34, 'name': 'snowfield', 'trainid': 31}, {'color': [255, 41, 10], 'id': 35, 'name': 'grandstand', 'trainid': 32}, {'color': [7, 255, 255], 'id': 36, 'name': 'sea', 'trainid': 33}, {'color': [224, 255, 8], 'id': 37, 'name': 'river', 'trainid': 34}, {'color': [255, 61, 6], 'id': 39, 'name': 'waterfall', 'trainid': 35}, {'color': [255, 194, 7], 'id': 40, 'name': 'water', 'trainid': 36}, {'color': [255, 122, 8], 'id': 41, 'name': 'billboard_or_Bulletin_Board', 'trainid': 37}, {'color': [0, 255, 20], 'id': 42, 'name': 'sculpture', 'trainid': 38}, {'color': [255, 5, 153], 'id': 44, 'name': 'flag', 'trainid': 39}, {'color': [6, 51, 255], 'id': 45, 'name': 'parasol_or_umbrella', 'trainid': 40}, {'color': [235, 12, 255], 'id': 46, 'name': 'cushion_or_carpet', 'trainid': 41}, {'color': [160, 150, 20], 'id': 47, 'name': 'tent', 'trainid': 42}, {'color': [0, 163, 255], 'id': 48, 'name': 'roadblock', 'trainid': 43}, {'color': [250, 10, 15], 'id': 50, 'name': 'bus', 'trainid': 44}, {'color': [20, 255, 0], 'id': 51, 'name': 'truck', 'trainid': 45}, {'color': [31, 255, 0], 'id': 52, 'name': 'bicycle', 'trainid': 46}, {'color': [255, 31, 0], 'id': 53, 'name': 'motorcycle', 'trainid': 47}, {'color': [255, 224, 0], 'id': 54, 'name': 'wheeled_machine', 'trainid': 48}, {'color': [153, 255, 0], 'id': 55, 'name': 'ship_or_boat', 'trainid': 49}, {'color': [0, 0, 255], 'id': 56, 'name': 'raft', 'trainid': 50}, {'color': [255, 71, 0], 'id': 57, 'name': 'airplane', 'trainid': 51}, {'color': [0, 235, 255], 'id': 58, 'name': 'tyre', 'trainid': 52}, {'color': [0, 173, 255], 'id': 59, 'name': 'traffic_light', 'trainid': 53}, {'color': [31, 0, 255], 'id': 60, 'name': 'lamp', 'trainid': 54}, {'color': [11, 200, 200], 'id': 61, 'name': 'person', 'trainid': 55}, {'color': [255, 82, 0], 'id': 62, 'name': 'cat', 'trainid': 56}, {'color': [0, 255, 112], 'id': 65, 'name': 'cattle', 'trainid': 57}, {'color': [0, 255, 133], 'id': 66, 'name': 'other_animal', 'trainid': 58}, {'color': [255, 0, 0], 'id': 67, 'name': 'tree', 'trainid': 59}, {'color': [255, 163, 0], 'id': 68, 'name': 'flower', 'trainid': 60}, {'color': [255, 102, 0], 'id': 69, 'name': 'other_plant', 'trainid': 61}, {'color': [194, 255, 0], 'id': 70, 'name': 'toy', 'trainid': 62}, {'color': [0, 143, 255], 'id': 71, 'name': 'ball_net', 'trainid': 63}, {'color': [51, 255, 0], 'id': 72, 'name': 'backboard', 'trainid': 64}, {'color': [0, 82, 255], 'id': 73, 'name': 'skateboard', 'trainid': 65}, {'color': [0, 255, 41], 'id': 74, 'name': 'bat', 'trainid': 66}, {'color': [0, 255, 173], 'id': 75, 'name': 'ball', 'trainid': 67}, {'color': [10, 0, 255], 'id': 76, 'name': 'cupboard_or_showcase_or_storage_rack', 'trainid': 68}, {'color': [173, 255, 0], 'id': 77, 'name': 'box', 'trainid': 69}, {'color': [0, 255, 153], 'id': 78, 'name': 'traveling_case_or_trolley_case', 'trainid': 70}, {'color': [255, 92, 0], 'id': 79, 'name': 'basket', 'trainid': 71}, {'color': [255, 0, 255], 'id': 80, 'name': 'bag_or_package', 'trainid': 72}, {'color': [255, 0, 245], 'id': 81, 'name': 'trash_can', 'trainid': 73}, {'color': [255, 173, 0], 'id': 83, 'name': 'plate', 'trainid': 74}, {'color': [255, 0, 20], 'id': 84, 'name': 'tub_or_bowl_or_pot', 'trainid': 75}, {'color': [255, 184, 184], 'id': 85, 'name': 'bottle_or_cup', 'trainid': 76}, {'color': [0, 31, 255], 'id': 86, 'name': 'barrel', 'trainid': 77}, {'color': [0, 255, 61], 'id': 87, 'name': 'fishbowl', 'trainid': 78}, {'color': [0, 71, 255], 'id': 88, 'name': 'bed', 'trainid': 79}, {'color': [255, 0, 204], 'id': 89, 'name': 'pillow', 'trainid': 80}, {'color': [0, 255, 194], 'id': 90, 'name': 'table_or_desk', 'trainid': 81}, {'color': [0, 255, 82], 'id': 91, 'name': 'chair_or_seat', 'trainid': 82}, {'color': [0, 10, 255], 'id': 92, 'name': 'bench', 'trainid': 83}, {'color': [0, 112, 255], 'id': 93, 'name': 'sofa', 'trainid': 84}, {'color': [51, 0, 255], 'id': 94, 'name': 'shelf', 'trainid': 85}, {'color': [0, 194, 255], 'id': 95, 'name': 'bathtub', 'trainid': 86}, {'color': [0, 255, 163], 'id': 97, 'name': 'commode', 'trainid': 87}, {'color': [255, 153, 0], 'id': 98, 'name': 'roaster', 'trainid': 88}, {'color': [0, 255, 10], 'id': 99, 'name': 'other_machine', 'trainid': 89}, {'color': [255, 112, 0], 'id': 100, 'name': 'refrigerator', 'trainid': 90}, {'color': [143, 255, 0], 'id': 101, 'name': 'washing_machine', 'trainid': 91}, {'color': [82, 0, 255], 'id': 102, 'name': 'Microwave_oven', 'trainid': 92}, {'color': [163, 255, 0], 'id': 103, 'name': 'fan', 'trainid': 93}, {'color': [255, 235, 0], 'id': 104, 'name': 'curtain', 'trainid': 94}, {'color': [8, 184, 170], 'id': 105, 'name': 'textiles', 'trainid': 95}, {'color': [133, 0, 255], 'id': 106, 'name': 'clothes', 'trainid': 96}, {'color': [0, 255, 92], 'id': 107, 'name': 'painting_or_poster', 'trainid': 97}, {'color': [184, 0, 255], 'id': 108, 'name': 'mirror', 'trainid': 98}, {'color': [255, 0, 31], 'id': 109, 'name': 'flower_pot_or_vase', 'trainid': 99}, {'color': [0, 184, 255], 'id': 110, 'name': 'clock', 'trainid': 100}, {'color': [0, 214, 255], 'id': 111, 'name': 'book', 'trainid': 101}, {'color': [92, 255, 0], 'id': 113, 'name': 'blackboard', 'trainid': 102}, {'color': [112, 224, 255], 'id': 115, 'name': 'screen_or_television', 'trainid': 103}, {'color': [163, 0, 255], 'id': 117, 'name': 'printer', 'trainid': 104}, {'color': [153, 0, 255], 'id': 118, 'name': 'Mobile_phone', 'trainid': 105}, {'color': [71, 255, 0], 'id': 119, 'name': 'keyboard', 'trainid': 106}, {'color': [255, 0, 163], 'id': 120, 'name': 'other_electronic_product', 'trainid': 107}, {'color': [255, 204, 0], 'id': 121, 'name': 'fruit', 'trainid': 108}, {'color': [255, 0, 143], 'id': 122, 'name': 'food', 'trainid': 109}, {'color': [0, 255, 235], 'id': 123, 'name': 'instrument', 'trainid': 110}, {'color': [133, 255, 0], 'id': 124, 'name': 'train', 'trainid': 111}]

id_map = {0:0}
test_list = [i for i in range(1,125) if i not in [3,7,11,22,38,44,53,60,79,86,99,103,116]]
for idx,i in enumerate(test_list):
    id_map[i] = idx+1
# id_map[255] = 255
    
# for cat in COCO_CATEGORIES_Seen:
#     id_map[cat["id"]] = cat["trainid"]
    
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

seen_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_seen.npy').tolist()
val_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_unseen.npy').tolist()
novel_cls = np.load(r'/home/lixinhao/vss/mmseg/handle_data/group_novel.npy').tolist()

train_class = seen_cls+val_cls

manager = Manager()
idset=manager.list()

def worker(file_tuple):
    read_path, out_path_file = file_tuple
    
    print("read: ",read_path,out_path_file)
    
    lab = np.asarray(Image.open(read_path))
    
    # for i in np.unique(lab):
    #     if i not in idset:
    #         idset.append(i)
                    
    assert lab.dtype == np.uint8

    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        # if obj_id in train_class:
        # test_list = [i for i in range(0,125) if i not in [3,7,11,22,38,44,53,60,79,86,99,103,116]]
        if obj_id <=111:
            output[lab == obj_id] = obj_id
            # output[lab == obj_id] = id_map[obj_id]
            
            # if obj_id not in idset:
            #     idset.append(obj_id)

    Image.fromarray(output).save(out_path_file)

if __name__ == "__main__":
    
    dataset_dir = Path() / "/home/lixinhao/vss/data/vspw/VSPW_480p/data" 
    
    pool = Pool(32)
    
    for t in ["train","val"]:
    # for t in ["val"]:
        file_list = []
        for root, dirs, files in os.walk(dataset_dir):
            for filename in files:
                # print(root,dirs,files)
                seri = os.path.split(root)[0].split(os.path.sep)[-1]
                type_d = os.path.split(root)[1]
                # print("seri",seri,type_d)
                if t == "train":
                    work_set = train_set
                else:
                    work_set = val_set
                if(seri in work_set and type_d == 'mask'):
                    # print(root,dirs,filename)
                    # print(root)
                    directory, s_filename = os.path.split(root)
                    # print(directory.split(os.path.sep)[-2:])
                    out_path = Path() / "/home/lixinhao/vss/data/vspw/VSPW_480p" / "Detectron" / (t +"_seen") / "/".join(directory.split(os.path.sep)[-2:]) 
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path_file = out_path/filename
                    read_path = Path() / root / filename
                    
                    file_list.append((read_path,out_path_file))
                    
        pool.map(worker,file_list)
        idset.sort()
        for i in idset:
            print(i)  
        print('done {}'.format(t))
        
        
         
             
        # lab = np.asarray(Image.open(read_path))
        
        # assert lab.dtype == np.uint8

        # output = np.zeros_like(lab, dtype=np.uint8) + 255
        # for obj_id in np.unique(lab):
        #     if obj_id in id_map:
        #         output[lab == obj_id] = id_map[obj_id]

        # Image.fromarray(output).save(out_path_file)