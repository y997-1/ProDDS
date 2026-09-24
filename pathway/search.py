import pandas as pd

# 读取CSV文件
file_path = './data/g_smiles.csv'  # 请将此路径替换为您的CSV文件路径
df = pd.read_csv(file_path)

# 查找包含特定值的行
search_value = 243
tolerance = 1e-9  # 定义一个非常小的阈值

# 创建一个布尔掩码，近似比较数值
mask = df.applymap(lambda x: abs(x - search_value) < tolerance if isinstance(x, (int, float)) else False)
result = df[mask.any(axis=1)]

# 打印查找结果
if not result.empty:
    print("找到包含值 {} 的行：".format(search_value))
    for index in result.index:
        print("行号:", index)
        print(df.iloc[index])
else:
    print("未找到包含值 {} 的行。".format(search_value))
