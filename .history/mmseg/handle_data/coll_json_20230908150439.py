import json

CLASSES = {"others": "0", "wall": "1", "ceiling": "2", "door": "3", "stair": "4", "ladder": "5", 
    "escalator": "6", "Playground_slide": "7", "handrail_or_fence": "8", "window": "9", 
    "rail": "10", "goal": "11", "pillar": "12", "pole": "13", "floor": "14",
    "ground": "15", "grass": "16", "sand": "17", "athletic_field": "18", "road": "19", "path": "20",
    "crosswalk": "21", "building": "22", "house": "23", "bridge": "24", "tower": "25", "windmill": "26",
    "well_or_well_lid": "27", "other_construction": "28", "sky": "29", "mountain": "30", "stone": "31",
    "wood": "32", "ice": "33", "snowfield": "34", "grandstand": "35", "sea": "36", "river": "37", 
    "lake": "38", "waterfall": "39", "water": "40", "billboard_or_Bulletin_Board": "41", "sculpture": "42",
    "pipeline": "43", "flag": "44", "parasol_or_umbrella": "45", "cushion_or_carpet": "46", "tent": "47",
    "roadblock": "48", "car": "49", "bus": "50", "truck": "51", "bicycle": "52", "motorcycle": "53",
    "wheeled_machine": "54", "ship_or_boat": "55", "raft": "56", "airplane": "57", "tyre": "58",
    "traffic_light": "59", "lamp": "60", "person": "61", "cat": "62", "dog": "63", "horse": "64",
    "cattle": "65", "other_animal": "66", "tree": "67", "flower": "68", "other_plant": "69", "toy": "70",
    "ball_net": "71", "backboard": "72", "skateboard": "73", "bat": "74", "ball": "75",
    "cupboard_or_showcase_or_storage_rack": "76", "box": "77", "traveling_case_or_trolley_case": "78",
    "basket": "79", "bag_or_package": "80", "trash_can": "81", "cage": "82", "plate": "83",
    "tub_or_bowl_or_pot": "84", "bottle_or_cup": "85", "barrel": "86", "fishbowl": "87", "bed": "88",
    "pillow": "89", "table_or_desk": "90", "chair_or_seat": "91", "bench": "92", "sofa": "93",
    "shelf": "94", "bathtub": "95", "gun": "96", "commode": "97", "roaster": "98", "other_machine": "99",
    "refrigerator": "100", "washing_machine": "101", "Microwave_oven": "102", "fan": "103", "curtain": "104",
    "textiles": "105", "clothes": "106", "painting_or_poster": "107", "mirror": "108", "flower_pot_or_vase": "109",
    "clock": "110", "book": "111", "tool": "112", "blackboard": "113", "tissue": "114", "screen_or_television": "115",
    "computer": "116", "printer": "117", "Mobile_phone": "118", "keyboard": "119", "other_electronic_product": "120",
    "fruit": "121", "food": "122", "instrument": "123", "train": "124"}

# 读取两个 JSON 文件
with open('file1.json', 'r') as file1, open('file2.json', 'r') as file2:
    data1 = json.load(file1)
    data2 = json.load(file2)

# 合并数据并转换为所需格式
merged_data = []
id_counter = 1  # 用于为每个条目分配唯一的ID

# 处理第一个文件的数据
for item in data1:
    merged_item = {
        "color": item["color"],
        "isthing": item["isthing"],
        "id": id_counter,
        "name": item["name"]
    }
    merged_data.append(merged_item)
    id_counter += 1

# 处理第二个文件的数据
for item in data2:
    merged_item = {
        "color": item["color"],
        "isthing": item["isthing"],
        "id": id_counter,
        "name": item["name"]
    }
    merged_data.append(merged_item)
    id_counter += 1

# 保存合并后的数据为 JSON 文件
with open('merged.json', 'w') as merged_file:
    json.dump(merged_data, merged_file, indent=4)
