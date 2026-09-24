# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# 数据点 (lr 和 AUC 的值)
lr_values = np.array([0.1, 0.01, 0.001, 0.0001])
auc_values = np.array([0.51, 0.72, 0.86, 0.60])

# 创建图形
plt.plot(lr_values, auc_values, marker='o', linestyle='-', color='black', label='AUC vs lr')


# 设置横纵坐标
plt.xlabel('lr', fontsize=12)
plt.ylabel('AUC', fontsize=12)

# 设置纵坐标范围
plt.ylim(0, 1)

# 设置横坐标刻度
plt.xticks(np.logspace(np.log10(0.0001), np.log10(0.1), 4))

# 不显示网格线
plt.grid(False)

# 显示图例
plt.legend()

# 显示图形
plt.show()

# 保存图像为文件
plt.savefig('lr')

# 关闭图像显示
plt.close()
