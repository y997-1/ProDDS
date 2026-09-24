# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F
import random
import os
import time
from process import *
from han_model import Model
from model import MLP_P
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Helper function to generate a boolean mask for training/testing
def bool_mask(length, indices):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[indices] = 1
    return mask

# Function to evaluate accuracy
def acc(pred, label, threshold=0.5):
    pred_label = (pred > threshold).float()
    return (pred_label == label).float().mean().item()

# Prediction function for new drug pairs
def predict_synergy_scores(valid_pairs, model, mlp, drug_feat, target_feat, p_feature, hetero_graph):
    model.eval()
    mlp.eval()
    
    # Construct node features dictionary
    p_feat = mlp(p_feature.to(torch.float32), 'relu')
    node_features = {'drug': drug_feat, 'target': target_feat, 'pathway': p_feat}
    
    # Extract drug1 and drug2 from valid pairs
    drug1 = torch.tensor([pair[0] for pair in valid_pairs], dtype=torch.int64)
    drug2 = torch.tensor([pair[1] for pair in valid_pairs], dtype=torch.int64)
    
    # Perform prediction for drug pairs
    with torch.no_grad():
        pred = model(hetero_graph, node_features, drug_feat.to(torch.float32), drug1, drug2)
    
    return pred

def main():
    start_time = time.time()  # Start timing the process
    
    # Load data (paths should be adjusted as needed)
    drug_data = pd.read_csv('./data/g_smiles.csv')
    drug_src = np.array(drug_data['g_id1'])
    drug_dst = np.array(drug_data['g_id2'])
    similar_label = np.array(drug_data['simi'])
    synergy_label = np.array(drug_data['class'])

    n_drug = 1493
    n_synergy = 30964
    n_target = 726
    n_pathway = 2201

    target_data = pd.read_csv('./data/drug_target.csv')
    target_list = np.array(target_data['t_id'])
    drug_list = np.array(target_data['g_id'])

    drug_feature = pd.read_csv('./data/drug_feature.csv')
    drug_feat = torch.tensor(np.array(drug_feature))
    drug_feat = torch.reshape(drug_feat, (-1, 300))

    target_target_data = pd.read_csv('./data/target_target_id.csv')
    target1 = np.array(target_target_data['t_id1'])
    target2 = np.array(target_target_data['t_id2'])

    target_feature = pd.read_csv('./data/target_feature.csv')
    normalize = nn.Softmax(dim=0)

    target_feat = torch.tensor(np.array(target_feature))
    target_feat = normalize(target_feat)

    target_feat = target_feat.to(torch.float32)
    target_feat = torch.reshape(target_feat, (-1, 300))

    target_pathway = pd.read_csv('./data/target_pathway_id.csv')
    target_id = np.array(target_pathway['t_id'])
    pathway_id = np.array(target_pathway['p_id'])

    pathway_feature = np.array(pd.read_csv('./data/p_feature.csv'))
    p_feature = torch.tensor(pathway_feature)
    p_feature = torch.reshape(p_feature, (-1, 9087))

    # Hetero graph
    hetero_graph = dgl.heterograph({
        ('drug', 'ddi', 'drug'): (drug_src, drug_dst),
        ('drug', 'ddi', 'drug'): (drug_dst, drug_src),
        ('drug', 'dds', 'drug'): (drug_src, drug_dst),
        ('drug', 'dds', 'drug'): (drug_dst, drug_src),
        ('drug', 'dt', 'target'): (drug_list, target_list),
        ('target', 'td', 'drug'): (target_list, drug_list),
        ('target', 'tt', 'target'): (target1, target2),
        ('target', 'tt', 'target'): (target2, target1),
        ('target', 'tp', 'pathway'): (target_id, pathway_id),
        ('pathway', 'pt', 'target'): (pathway_id, target_id),
    })

    hetero_graph.edges['dds'].data['label'] = torch.tensor(synergy_label)
    hetero_graph.edges['ddi'].data['label'] = torch.tensor(similar_label)

    hetero_graph.nodes['drug'].data['feature'] = drug_feat
    hetero_graph.nodes['target'].data['feature'] = target_feat

    # Model
    meta_paths = [['dt','td'], ['dt', 'tt', 'td'], ['dt', 'tp', 'pt', 'td']]
    model = Model(meta_paths, 300, 256, 128, 64, 32, 16, 4, 1, hetero_graph.etypes)
    mlp = MLP_P(9087, 300)

    # Load pre-trained model (if needed)
    model_save_path = './saved_models'
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'model_epoch_99.pth')))
    mlp.load_state_dict(torch.load(os.path.join(model_save_path, 'mlp_epoch_99.pth')))

    # New drug pairs for prediction (valid_pairs should not be in the training set)
    valid_pairs = [(random.randint(0, n_drug-1), random.randint(0, n_drug-1)) for _ in range(100)]  # Example new pairs
    
    # Predict synergy scores for new drug pairs
    predicted_scores = predict_synergy_scores(valid_pairs, model, mlp, drug_feat, target_feat, p_feature, hetero_graph)

    # Prepare the results for saving
    result_df = pd.DataFrame({
        'Drug1': [pair[0] for pair in valid_pairs],
        'Drug2': [pair[1] for pair in valid_pairs],
        'Predicted_Synergy_Score': predicted_scores.numpy().flatten()  # Convert tensor to numpy and flatten
    })

    # Save the results to a CSV file
    result_file_path = './predicted_synergy_scores.csv'
    result_df.to_csv(result_file_path, index=False)
    print(f"Predicted synergy scores saved to {result_file_path}")

    end_time = time.time()
    print(f'Total execution time: {end_time - start_time:.2f} seconds')

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
