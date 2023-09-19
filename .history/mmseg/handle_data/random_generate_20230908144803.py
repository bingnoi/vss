import numpy as np

# 定义总数和随机数的数量
total_sum = 125
num_numbers = 3

# 定义三组数的比例
ratios = [0.3, 0.4, 0.3]  # 这里的比例可以根据你的需求进行调整

# 生成三组各不相同的随机数
random_numbers = []
for ratio in ratios:
    num = int(total_sum * ratio)
    if num > 0:
        # 生成0到124范围内的随机数
        random_nums = np.random.randint(0, 125, size=num)
        random_numbers.extend(random_nums)

# 如果总数不够，补充到总数为125
while len(random_numbers) < total_sum:
    remaining_sum = total_sum - sum(random_numbers)
    random_num = np.random.randint(0, 125 - remaining_sum)
    random_numbers.append(random_num)

# 打乱随机数的顺序
np.random.shuffle(random_numbers)

# 分别保存三组随机数到不同的文件
for i, numbers in enumerate(random_numbers):
    file_name = f'random_numbers_{i + 1}.npy'
    np.save(file_name, numbers)
    print(f'Saved {len(random_numbers)} random numbers to {file_name}')

# 查看生成的三组随机数
for i, numbers in enumerate(random_numbers):
    print(f'Random Numbers {i + 1}: {numbers}')
