from rdkit import Chem

from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import QED
import csv
import pandas as pd
import numpy as np

def smiles_to_ecfp(smiles_list):
    fps = []

    for x in range(len(smiles_list)):

        mol = Chem.MolFromSmiles(str(smiles_list[x]))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=300)
        fps.append(fp)

    file = open('./data/title.csv', 'r')

    result = list()
    for c in file.readlines():
        c_array = c.split(",")
        result.append(c_array)

    with open('./Model/drug_feature.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result[0])


    for i in range(len(fps)):

        X_feature = list(fps[i])

        with open('./Model/drug_feature.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(X_feature)



data_smiles = pd.read_csv('./Model/drug-cid-smiles.xlsx')
smiles_list = np.array(data_smiles['SMILES'])
smiles_to_ecfp(smiles_list)

