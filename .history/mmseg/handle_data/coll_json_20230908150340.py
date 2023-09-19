import json

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
