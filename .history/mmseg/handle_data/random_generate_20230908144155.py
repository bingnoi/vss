import numpy as np

# 定义总数和随机数的数量
total_sum = 124
num_numbers = 3

# 生成三组各不相同的随机数，使其总和等于total_sum
random_numbers = []
for _ in range(num_numbers - 1):
    # 生成随机数，范围在1到total_sum
    random_num = np.random.randint(1, total_sum - sum(random_numbers) - (num_numbers - len(random_numbers)) + 1)
    random_numbers.append(random_num)

# 最后一组随机数等于总数减去前两组的和
random_numbers.append(total_sum - sum(random_numbers))

# 打乱随机数的顺序
np.random.shuffle(random_numbers)

# 分别保存三组随机数到不同的文件
for i, numbers in enumerate(random_numbers):
    file_name = f'random_numbers_{i + 1}.npy'
    np.save(file_name, numbers)
    print(f'Saved {len(numbers)} random numbers to {file_name}')
