import matplotlib
# 使用 'Agg' 后端以在无图形界面的服务器上保存图像
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 示例数据（请替换成自己的真实数据）
x_vals = [0.24, 0.5, 0.61, 0.72, 0.82]
y_vals = [0.23, 0.13, 0.05, 0.04, 0.03]

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(5, 4))

# 绘制散点
ax.scatter(x_vals, y_vals, color='black')

# 绘制 y=1 的虚线
ax.axhline(y=1, color='black', linestyle='dotted')

# 设置坐标轴范围（可根据需要调整）
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 2.0)

# 设置坐标轴刻度和标签
ax.set_xlabel('Effect', fontsize=12)
ax.set_ylabel('CI', fontsize=12)

# 隐藏网格、去除右边和上边边框
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='in')

# 自动调整边距
plt.tight_layout()

# 将图像保存为 PNG 文件，DPI 可以根据需求调整
plt.savefig('CMCI.png', dpi=300)



