from squences_feature import *
import pandas as pd
import numpy as np
import glob
import os
import csv

path = r'D:\1.Deep learning\meta path for drug synergy\target_sequence'
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
dl = []


for f in file:
    print(f)
    data = []
    for line in open(f):
        data.append(line)

    feature = get_feature(str(data))
    #print(feature)

    with open('D:/1.Deep learning/meta path for drug synergy/target_sequence/target_feature.csv', 'a', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerow([str(f), np.array(feature)])

