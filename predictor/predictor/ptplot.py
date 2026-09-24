# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
y_pred_path = '/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/output/cv_2503172020/y_preds.pkl'
y_true_path = '/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/output/cv_2503230953/y_true.pkl'


with open(y_pred_path, 'rb') as f:
    y_pred = pickle.load(f)

with open(y_true_path, 'rb') as f:
    y_true = pickle.load(f)

# 将列表转换为numpy数组，方便后续计算和绘图
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 筛选出横纵坐标均在-100到100之间的数据点
mask = (y_true >= -100) & (y_true <= 100) & (y_pred >= -100) & (y_pred <= 100)
y_true_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_true_filtered, y_pred_filtered, color='#a7c7e7', alpha=0.6, label='Data points')

# 拟合线性回归线
slope, intercept = np.polyfit(y_true_filtered, y_pred_filtered, 1)
plt.plot(y_true_filtered, slope * y_true_filtered + intercept, color='#004a7c', label=f'y={slope:.2f}x+{intercept:.2f}')

# 添加标签和标题
plt.xlabel('Predicted Synergy Scores')
plt.ylabel('Ground Truth')

# 设置显示范围
plt.xlim(-100, 100)
plt.ylim(-100, 100)


plt.legend()

# 保存图像为文件
plt.savefig('GATscore_scatter_plot.png')

# 关闭图像显示
plt.close()




