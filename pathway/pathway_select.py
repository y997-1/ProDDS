import pandas as pd
import numpy as np



def pathway_select(name, src, dst):

    #drug_drug = pd.read_csv('./data/')
    drug_target = pd.read_csv('./data/drug_target_name.csv')
    target_target = pd.read_csv('./data/target_target.csv')
    target_pathway = pd.read_csv('./data/target_pathway_id.csv')

    #drugA = np.array(drug_drug['drugA'])
    #drugB = np.array(drug_drug['drugB'])
    DT_drug = np.array(drug_target['drug_name'])
    DT_target = np.array(drug_target['Target'])
    TT_target1 = np.array(target_target['target1'])
    TT_target2 = np.array(target_target['target2'])
    TP_target = np.array(target_pathway['Target'])
    TP_pathway = np.array(target_pathway['Pathway'])

    output = []

    if name == 'DTD':
        for i in range(len(DT_drug)):
            if DT_drug[i] == str(src):
                target = DT_target[i]
                for j in range(len(DT_target)):
                    if DT_target[j] == target:
                        if DT_drug[j] == str(dst):
                            output.append([src,target,dst])

        print('Search results total: ',len(output))

    elif name == 'DTTD':
        for i in range(len(DT_drug)):
            if DT_drug[i] == str(src):
                target = DT_target[i]
                for j in range(len(TT_target1)):
                    if TT_target1[j] == target:
                        new_target = TT_target2[j]
                        for k in range(len(DT_target)):
                            if new_target == DT_target[k]:
                                if DT_drug[k] == str(dst):
                                    output.append([src, target, new_target, dst])

        print('Search results total: ',len(output))

    elif name == 'DTPTD':
        for i in range(len(DT_drug)):
            if DT_drug[i] == str(src):
                target = DT_target[i]
                for j in range(len(TP_target)):
                    if TP_target[j] == target:
                        pathway = TP_pathway[j]
                        for k in range(len(TP_pathway)):
                            if pathway == TP_pathway[k]:
                                new_target = TP_target[k]
                                for l in range(len(DT_target)):
                                    if DT_target[l] == new_target:
                                        if DT_drug[l] == str(dst):
                                            output.append([src, target, pathway, new_target, dst])
        print('Search results total: ',len(output))

    return output


#DTD_list = pathway_select('DTD','Idelalisib', 'Duvelisib')
#print(DTD_list)
DTTD_list = pathway_select('DTTD','Gefitinib', 'Afatinib')
print(DTTD_list)
#DTPTD_list = pathway_select('DTPTD','Afatinib', 'Gefitinib')
#print(DTPTD_list)



