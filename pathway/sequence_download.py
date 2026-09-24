import urllib.request
import requests
import pandas as pd
import numpy as np


target_all = pd.read_csv('t_id.csv')
target_list = np.array(target_all['Target'])


session = requests.Session()

for i in range(len(target_list)):
    try:
        #print(target_list[i])
        response = urllib.request.urlopen('https://rest.uniprot.org/uniprotkb/'+str(target_list[i])+'.txt')
        sequence = response.read()
        #print(sequence)

        with open(str(target_list[i])+'.txt','ab') as f:
            f.write(sequence)
            f.close()

        print(i)
    except:
        print('%i error！'%i)
