import numpy as np

# 定义总数和随机数的数量
total_sum = 124
num_numbers = 3

# 定义三组数的比例
ratios = [0.3, 0.4, 0.3]  # 这里的比例可以根据你的需求进行调整

# 生成三组各不相同的随机数
random_numbers = []
for ratio in ratios:
    num = int(total_sum * ratio)
    if num > 0:
        # 生成0到124范围内的不重复随机数
        random_nums = np.random.choice(125, size=num, replace=False)
        random_numbers.extend(random_nums)

# 打印生成的三组随机数
for i, numbers in enumerate(random_numbers):
    print(f'Random Numbers {i + 1}: {numbers}')
