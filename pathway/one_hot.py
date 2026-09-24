import pandas as pd
import numpy as np
import csv


target_list = pd.read_csv('target.csv')
#data_all = pd.read_csv('data.csv')
#pathway_feature = pd.read_csv('pathway.csv')

target_id = np.array(target_list['target'])
#print(len(pathway_feature))

csv_file=open('pathway.csv')
csv_reader_lines = csv.reader(csv_file)
time = 0

for one_line in csv_reader_lines:
    dim_1 = [0 for index in range(len(target_id))]
    #print(len(dim_1))

    for i in range(len(one_line)):
        for j in range(len(target_id)):
            if one_line[i] == target_id[j]:
                dim_1[j] = 1
    #print(dim_1)

    for k in range(len(dim_1)):
        with open('one_hot.txt','a') as f:
            f.write(str(dim_1[k]))
            f.write(' ')
            if k == len(dim_1) - 1:
                f.write('\n')
    time += 1
    print(time)





