# ProDDS: A two-stage graph-based integration paradigm (Pre-ProDDS/ProDDS) for cell-line-specific anticancer drug synergy prediction incorporating protein-protein interaction networks


## Environment Setup

pandas==1.1.1
joblib==0.17.0
dgl==0.6.1
matplotlib==3.3.1
numpy==1.19.5
torch==1.7.0
scikit_learn==0.24.2

## Run

# Process pathway data
cd pathway
python data_preprocessing.py
cd ..

# Extract drug features
cd drug
python gen_feat.py
cd ..

# Extract cell line features
cd cell
python GATgen_feat.py
cd ..

cd predictor

# 5-fold nested cross-validation
python cross_validation.py --epoch 500 --batch 512 --hidden 4096 --lr 0.0001
~~~


## Cite

> Xiaowen Wang, Hongming Zhu, Yizhi Jiang, Yulong Li, Chen Tang, Xiaohan Chen, Yunjie Li, Qi Liu, Qin Liu, PRODeepSyn: predicting anticancer synergistic drug combinations by embedding cell lines with protein–protein interaction network, Briefings in Bioinformatics, Volume 23, Issue 2, March 2022, bbab587, https://doi.org/10.1093/bib/bbab587

