# -*- coding: utf-8 -*-
from rdkit import Chem
import pandas as pd
import numpy as np
import csv
import os
import networkx as nx
import networkx as nx
from rdkit import Chem
from karateclub import Graph2Vec


def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atom_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))
    return G
    
def smiles_to_graph2vec(smiles_list):
    graphs = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            g = mol_to_nx(mol)
            graphs.append(g)
        except:
            continue

    # 使用 Graph2Vec 训练嵌入模型
    model = Graph2Vec(dimensions=300)  # 输出维度与原始ECFP一致
    model.fit(graphs)
    embeddings = model.get_embedding()

    # 读取列名标题（与原格式一致）
    with open('./data/title.csv', 'r') as file:
        result = [line.strip().split(',') for line in file.readlines()]

    # 写入特征文件
    os.makedirs('./Model', exist_ok=True)
    with open('./Model/drug_feature.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result[0])
        for row in embeddings:
            writer.writerow(row)

# 主程序入口
data_smiles = pd.read_excel('./Model/drug-cid-smiles.xlsx')
smiles_list = np.array(data_smiles['SMILES'])
smiles_to_graph2vec(smiles_list)

