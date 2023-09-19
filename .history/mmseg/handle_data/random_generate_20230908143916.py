import numpy as np

# 生成8个不同的随机整数（范围可以根据需要调整）
random_numbers = np.random.choice(np.arange(1, 101), size=8, replace=False)

# 打印生成的随机数
print(random_numbers)

# 保存随机数到文件
np.save('seen_cls.npy', random_numbers)