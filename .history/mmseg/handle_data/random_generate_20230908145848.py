import numpy as np

# 定义总数量和三组数字的数量
total_count = 125
group_count = 3

# 定义每组数字的比例（可以根据需要调整）
group_ratios = [0.8, 0.3, 0.3]

# 计算每组数字的数量
group_sizes = [int(total_count * ratio) for ratio in group_ratios]

# 确保总数与要求的总数量相等
group_sizes[-1] += total_count - sum(group_sizes)

# 生成0到125的数字列表
all_numbers = list(range(total_count))

# 随机打乱数字顺序
np.random.shuffle(all_numbers)

# 初始化每组数字的起始和结束索引
start_idx = 0
groups = []

# 将数字按比例分配到每组
for size in group_sizes:
    end_idx = start_idx + size
    group = all_numbers[start_idx:end_idx]
    groups.append(group)
    start_idx = end_idx

# 打印每组的信息和具体内容
for i, group in enumerate(groups):
    file_name = f"group_{i}.npy"
    np.save(file_name, group)

    # 打印每组的信息
    print(f"Group {i}: Saved {len(group)} numbers to {file_name}")
    print(f"Group {i} Content: {group}")
