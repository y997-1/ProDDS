from squences_feature import *
import pandas as pd
import numpy as np
import glob
import os
import csv

path = r'D:\1.Deep learning\meta path for drug synergy\target\target_target'
'''
data_list = pd.read_csv('D:/1.Deep learning/meta path for drug synergy/data/t_id.csv')
target_id = np.array(data_list['Target'])

list_target = []

for f in os.listdir(path):
    #print(f)
    for i in range(len(target_id)):
        name = target_id[i] + '.txt'
        #print(name)
        if name == f:
            print(i)
            list_target.append(i)
print(list_target)
'''
file = glob.glob(os.path.join(path, "*.txt"))

#print(file)


for f in file:
    print(f)
    data = open(f)
    lines = data.readlines()
    #print(lines)

    for i in range(len(lines)):
        #print(line)

        if lines[i].startswith('CC   -!- INTERACTION:'):
            for j in range(100):
                i += 1
                with open('D:/1.Deep learning/meta path for drug synergy/target/target_target.txt', 'a',
                          newline='') as new_file:

                    new_file.write(lines[i])               
                

                if lines[i].startswith('CC   -!-'):
                    break

