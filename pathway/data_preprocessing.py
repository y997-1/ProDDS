'''
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 读取CSV文件
df = pd.read_csv('./data/g_smiles.csv')

# 将simi列中的数值保留到小数点后三位，并填补不够三位的小数
df['simi'] = df['simi'].apply(lambda x: f"{x:.3f}")

# 删除重复行
df.drop_duplicates(inplace=True)

# 删除包含缺失值的行
df.dropna(inplace=True)

# 转换SMILES到分子对象并标准化
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol if mol is not None else None

def standardize_smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None

# 应用函数
df['mol'] = df['smiles1'].apply(smiles_to_mol)
df['smiles1'] = df['mol'].apply(standardize_smiles)
df.drop('mol', axis=1, inplace=True)

# 保存处理后的数据
output_file_path = './data/processed_ggg.csv'  # 请将此路径替换为您想保存的文件路径
df.to_csv(output_file_path, index=False)

print("处理完成，结果已保存到", output_file_path)
'''

import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('./data/drug_target.csv')

# 删除重复项所在的行
data.drop_duplicates(inplace=True)

# 删除缺失值所在的行
data.dropna(inplace=True)

# 确保数据类型正确
data['drugbank_id'] = data['drugbank_id'].astype(str)
data['g_id'] = data['g_id'].astype(int)
data['Target'] = data['Target'].astype(str)
data['t_id'] = data['t_id'].astype(int)

# 检查异常值（以'g_id'和't_id'为例，假设它们的值应该在合理范围内）
g_id_min, g_id_max = 0, 100000  # 根据实际数据的合理范围设定
t_id_min, t_id_max = 0, 10000   # 根据实际数据的合理范围设定

# 剔除异常值
data = data[(data['g_id'] >= g_id_min) & (data['g_id'] <= g_id_max)]
data = data[(data['t_id'] >= t_id_min) & (data['t_id'] <= t_id_max)]

# 保存处理后的数据到新的 CSV 文件
output_file_path = './data/drug_target_cleaned.csv'
data.to_csv(output_file_path, index=False)

print(f"数据预处理完成，处理后的数据保存在 {output_file_path}")
