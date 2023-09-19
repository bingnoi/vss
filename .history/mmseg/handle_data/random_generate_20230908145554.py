import numpy as np

# 定义总数量和三组数字的数量
total_count = 125
group_count = 3

# 生成0到125的数字列表
all_numbers = list(range(total_count))

# 随机打乱数字顺序
np.random.shuffle(all_numbers)

# 将数字分成三组
groups = np.array_split(all_numbers, group_count,)

# 将每组数字保存到不同的文件
for i, group in enumerate(groups):
    file_name = f"group_{i}.npy"
    np.save(file_name, group)

    # 打印每组的信息
    print(f"Group {i}: Saved {len(group)} numbers to {file_name}")
