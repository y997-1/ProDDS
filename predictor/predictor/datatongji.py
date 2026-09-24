# -*- coding: utf-8 -*-
import pandas as pd
import warnings

# 忽略时区路径警告
warnings.filterwarnings("ignore", category=UserWarning, message="InvalidTZPathWarning")

# 读取CSV文件
file_path = '/public/home/Yule1/Desktop/DeepLearning/PPI/drug/labels.csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# 根据synergy列的值判断正样本和负样本
data['label'] = data['synergy'].apply(lambda x: 'positive' if x > 0 else 'negative')

# 统计不同细胞系上正样本和负样本的组数
group_counts = data.groupby(['cell_line', 'label']).size().unstack(fill_value=0)

# 将结果保存为CSV文件
output_file = 'group_counts_result.csv'  # 保存路径，可以修改为你需要的路径
group_counts.to_csv(output_file)

# 打印输出文件路径
print(f"Results have been saved to {output_file}")
