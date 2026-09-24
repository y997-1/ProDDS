import argparse
import os
import torch
import numpy as np
import pickle
import pandas as pd  # 导入 pandas 用于生成 CSV 文件

from datetime import datetime
from model.datasets import NOFastSynergyDataset, FastTensorDataLoader
from model.models import CNN
from model.utils import find_best_model
from const import DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, OUTPUT_DIR, SYNERGY_FILE

time_str = str(datetime.now().strftime('%y%m%d%H%M'))


def create_model(data, hidden_size, gpu_id=None):
    model = CNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model


def predict_no_labels(out_dir):
    n_folds = 5
    n_delimiter = 60
    y_preds = []
    raw_samples = []

    for test_fold in range(n_folds):
        test_data = NOFastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
                                         SYNERGY_FILE, use_folds=[test_fold], train=False)  # 使用空字符串或默认路径
        test_mdl_dir = os.path.join(out_dir, str(test_fold))
        try:
            model = create_model(test_data, 4096, None)
            model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
        except Exception:
            try:
                model = create_model(test_data, 8192, None)
                model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
            except Exception:
                model = create_model(test_data, 2048, None)
                model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))

        test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for drug1_feats, drug2_feats, cell_feats, _ in test_loader:  # `_` 表示忽略协同作用分数
                yp1 = model(drug1_feats, drug2_feats, cell_feats)
                yp2 = model(drug2_feats, drug1_feats, cell_feats)
                y_pred = (yp1 + yp2) / 2
                y_pred = y_pred.numpy().flatten()
                fold_preds.extend(y_pred)

        y_preds.extend(fold_preds)
        raw_samples.extend(test_data.raw_samples)  # 保留原始样本信息
        print("Predictions for fold {} saved.".format(test_fold))
        print("*" * n_delimiter + '\n')

    # 保存预测结果和原始样本为 CSV 文件
    csv_file = os.path.join(out_dir, 'predictions.csv')

    # 生成一个 DataFrame
    df = pd.DataFrame({
        "Drug1_ID": [sample[0] for sample in raw_samples],
        "Drug2_ID": [sample[1] for sample in raw_samples],
        "Cell_ID": [sample[2] for sample in raw_samples],
        "Predicted_Synergy_Score": y_preds  # 保存预测的协同作用分数
    })

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_file, index=False)
    print(f"Predictions saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mdl_dir', type=str, help="model dir")
    args = parser.parse_args()
    mdl_dir = os.path.join(OUTPUT_DIR, args.mdl_dir)
    predict_no_labels(mdl_dir)


if __name__ == '__main__':
    main()
