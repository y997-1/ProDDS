import pickle
import pandas as pd
import numpy as np

# 加载 pkl 文件
with open('/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/output/cv_2412041150/y_preds.pkl', 'rb') as f:
    data = pickle.load(f)

# 获取最长的数组长度
max_len = max(len(arr) for arr in data)

# 填充每个数组，确保它们的长度相同
data_padded = [np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan) for arr in data]

# 将数据转为一个 DataFrame，每个数组作为 DataFrame 的一列
df = pd.DataFrame(np.array(data_padded).T)  # .T 是转置操作，将每个数组变为一列

# 保存为 CSV 文件
df.to_csv('y_preds.csv', index=False)

print("转换完成，CSV 文件已保存。")
