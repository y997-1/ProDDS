# -*- coding: utf-8 -*-
from rdkit import Chem
import pandas as pd
import numpy as np
import csv
import warnings

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

warnings.filterwarnings("ignore")

def smiles_to_mol2vec(smiles_list, model_path='./model_300dim.pkl'):
    model = word2vec.Word2Vec.load(model_path)
    fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        sentence = mol2alt_sentence(mol, radius=1)
        vec = DfVec(sentences2vec([sentence], model, unseen='UNK')[0])
        fps.append(vec.vec)  # shape: (300,)

    # 加载列标题（假设和原来一样）
    with open('./data/title.csv', 'r') as file:
        result = [line.strip().split(',') for line in file.readlines()]
    
    with open('./Model/drug_feature.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result[0])
        for fp in fps:
            writer.writerow(fp)

# 主执行逻辑
data_smiles = pd.read_excel('./Model/drug-cid-smiles.xlsx')
smiles_list = np.array(data_smiles['SMILES'])
smiles_to_mol2vec(smiles_list)

