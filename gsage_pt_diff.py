import os
import sys
import random
import csv
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import SAGEConv

from transformers import get_scheduler

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold

import wandb


sweep_config = {
    'method': 'grid',
    'metric': {'name': 'mean_test_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [1e-04]},
        'weight_decay': {'values': [1e-05]},
        'epochs': {'values': [150]},
        'hidden_channels': {'values': [64]},
        'edge_threshold_percentile': {'values': [90]},
        'dropout': {'values': [0.0]},
        'diffusion_epoch': {'values': [2000]},
        'drop_prob': {'values': [0.5]},
        'guidance_scale': {'values': [10.5]},
        'timesteps': {'values': [200]},
        'emb': {'values': [32]},
        'vae_epochs': {'values': [300]},
        'sw_loss': {'values': [0]},
    }
}

device = 'cuda:2'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_augmented_abide_data(seed, abide_path, epoch, edge_threshold_percentile):
    """
    Load augmented ABIDE graphs from npz and convert to PyG Data.
    Ensures edge_attr shape (E,1) and adds self-loops if needed.
    """
    print(abide_path)
    data_list = []

    file_path = os.path.join(
        abide_path,
        f"diff_cond_seed_100/generated_data/all_matched_samples_seed_100_epoch_{epoch}.npz"
    )
    data = np.load(file_path, allow_pickle=True)
    node_features = data['node_features']
    labels = data['labels']

    for n in range(len(node_features)):
        node_feature = np.asarray(node_features[n])
        N = node_feature.shape[0]

        i, j = np.triu_indices(N, k=1)
        abs_mat = np.abs(node_feature)
        nz_vals = abs_mat[i, j][abs_mat[i, j] != 0]

        if nz_vals.size > 0:
            threshold = np.percentile(nz_vals, edge_threshold_percentile)
        else:
            threshold = np.inf

        upper_tri = np.zeros_like(node_feature, dtype=bool)
        if nz_vals.size > 0:
            upper_tri[i, j] = abs_mat[i, j] > threshold
        symmetric_adj = upper_tri | upper_tri.T

        rows0, cols0 = np.where(symmetric_adj)
        w0 = node_feature[rows0, cols0].astype(np.float32)

        self_loop = np.arange(N, dtype=np.int64)
        rows = np.concatenate([rows0, self_loop])
        cols = np.concatenate([cols0, self_loop])
        w = np.concatenate([w0, np.ones_like(self_loop, dtype=np.float32)])

        edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long, device=device)
        edge_attr = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(-1)
        x = torch.tensor(node_feature, dtype=torch.float32, device=device)
        y = torch.tensor(labels[n], dtype=torch.long, device=device)

        try:
            d = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(d)
        except Exception as e:
            print(f"[Aug] Error processing sample {n}: {e}")
            continue

    return data_list


def load_abide_list(data_type, fold_idx=0, num_folds=10, seed=100, site_filter="NYU"):
    data_dir = '/yourpath'
    phenotype_path = '/yourpath'

    df = pd.read_csv(phenotype_path)
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip()
    df = df.set_index('FILE_ID', drop=False)

    if site_filter is not None:
        df = df[df['SITE_ID'] == site_filter]

    all_files = [f for f in os.listdir(data_dir) if f.endswith('_rois_aal.1D')]
    valid_files, labels = [], []
    for filename in all_files:
        file_id = filename.replace('_rois_aal.1D', '').strip()
        if file_id in df.index:
            valid_files.append(filename)
            labels.append(df.loc[file_id, 'DX_GROUP'])

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_splits = list(skf.split(valid_files, labels))
    train_idx, test_idx = all_splits[fold_idx]

    if data_type == "train":
        selected_files = [valid_files[i] for i in train_idx]
    elif data_type == "test":
        selected_files = [valid_files[i] for i in test_idx]
    else:
        raise ValueError("data_type은 'train' 또는 'test'여야 합니다.")

    print(f"[{site_filter}] Fold {fold_idx+1}/{num_folds}, {data_type} set: {len(selected_files)} files")
    return np.array(selected_files)


class GraphDataset(Dataset):
    def __init__(self, data_filenames, device, edge_threshold_percentile):
        super(GraphDataset, self).__init__()
        self.abide_path = '/yourpath'
        self.phenotype_path = '/yourpath'
        self.device = device

        self.df = pd.read_csv(self.phenotype_path)
        self.df['FILE_ID'] = self.df['FILE_ID'].astype(str).str.strip()
        self.df = self.df.set_index('FILE_ID', drop=False)
        self.site_to_num = {site: i for i, site in enumerate(self.df['SITE_ID'].unique())}
        self.df['SITE_NUM'] = self.df['SITE_ID'].map(self.site_to_num)

        self.data_filenames = data_filenames
        self.edge_threshold_percentile = edge_threshold_percentile
        self.data_list = self._load_data()

    def _load_data(self):
        data_list = []
        p = self.edge_threshold_percentile
        for filename in self.data_filenames:
            file_path = os.path.join(self.abide_path, filename)
            file_id = filename.replace('_rois_aal.1D', '').strip()
            if file_id not in self.df.index:
                print(f"[WARNING] File {file_id} not in phenotype CSV")
                continue

            roi_signal = np.loadtxt(file_path)
            if np.all(roi_signal != 0):
                corr = np.corrcoef(roi_signal, rowvar=False)
                corr = np.arctanh(np.clip(corr, -0.9999, 0.9999))
                np.fill_diagonal(corr, 1.0)

                i, j = np.triu_indices(corr.shape[0], k=1)
                vals = np.abs(corr)[i, j]
                vals = vals[vals != 0]
                thr = np.percentile(vals, p) if vals.size > 0 else 0.0

                upper_tri = np.zeros_like(corr, dtype=bool)
                upper_tri[i, j] = np.abs(corr)[i, j] > thr
                A = upper_tri | upper_tri.T

                rows0, cols0 = np.where(A)
                w0 = corr[rows0, cols0]
                self_loops = np.arange(corr.shape[0])
                rows = np.concatenate([rows0, self_loops])
                cols = np.concatenate([cols0, self_loops])
                w = np.concatenate([w0, np.ones_like(self_loops, dtype=corr.dtype)])

                edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long, device=self.device)
                edge_attr = torch.tensor(w, dtype=torch.float, device=self.device).unsqueeze(-1)
                x = torch.tensor(corr, dtype=torch.float, device=self.device)

                label = int(self.df.loc[file_id, 'DX_GROUP'] == 1)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor(label, dtype=torch.long, device=self.device)
                )
                data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class GraphSAGE(nn.Module):
    def __init__(self, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = SAGEConv(116, 116)
        self.conv2 = SAGEConv(116, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels * 116, out_features=116 * 32)
        self.linear2 = nn.Linear(116 * 32, 2)
        self.layer_norm1 = nn.LayerNorm(116 * 32)
        self.mish = nn.Mish()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.mish(x)
        x = self.conv2(x, edge_index)
        x = self.mish(x)
        x = x.reshape(-1, 116 * self.hidden_channels)
        x = self.linear1(x)
        x = self.layer_norm1(x)
        x = self.mish(x)
        x = self.linear2(x)
        return x


def count_labels_in_data(train_loader, test_loader):
    train_counts = Counter()
    test_counts = Counter()
    for data in train_loader:
        train_counts.update(data.y.cpu().numpy())
    for data in test_loader:
        test_counts.update(data.y.cpu().numpy())
    return train_counts, test_counts


def evaluate(model, test_loader):
    model.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            with torch.cuda.amp.autocast():
                out = model(data.x, data.edge_index)
                probs = F.softmax(out, dim=1)[:, 1]
                preds = out.argmax(dim=1)
            y_true_list.append(data.y.detach().cpu().numpy())
            y_pred_list.append(preds.detach().cpu().numpy())
            y_prob_list.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_probs = np.concatenate(y_prob_list, axis=0)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = float('nan')

    print(
        f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, Sensitivity: {sensitivity:.4f}, "
        f"Specificity: {specificity:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'auc': auc
    }


def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, epochs):
    all_labels = []
    for data in train_loader:
        all_labels.extend(data.y.tolist())
    counts = Counter(all_labels)
    total = sum(counts.values())
    weights = torch.tensor(
        [1 - (counts.get(0, 0) / total), 1 - (counts.get(1, 0) / total)],
        device=device,
        dtype=torch.float
    )
    print("class weights:", weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        ep_pred, ep_label = [], []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            ep_pred.extend(out.argmax(dim=1).detach().cpu().numpy())
            ep_label.extend(data.y.cpu().numpy())

        tn, fp, fn, tp = confusion_matrix(ep_label, ep_pred).ravel()
        acc = (tp + tn) / max((tp + tn + fp + fn), 1)
        sens = tp / max((tp + fn), 1)
        spec = tn / max((tn + fp), 1)
        print(
            f"epoch: {epoch} epoch_loss: {epoch_loss:.4f} "
            f"accuracy: {acc:.4f} sensitivity: {sens:.4f} specificity: {spec:.4f}"
        )
        scheduler.step()

    return epoch_loss, evaluate(model, test_loader)


def train_and_evaluate_all_seeds_and_folds():
    wandb.init()
    cfg = wandb.config

    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay
    epochs = cfg.epochs
    hidden_channels = cfg.hidden_channels
    edge_threshold_percentile = cfg.edge_threshold_percentile
    diffusion_epoch = cfg.diffusion_epoch
    drop_prob = cfg.drop_prob
    guidance_scale = cfg.guidance_scale
    timestep = cfg.timesteps
    emb = cfg.emb
    vae = cfg.vae_epochs
    sw_loss = cfg.sw_loss

    num_folds = 5
    fold_results = []

    for fold_idx in range(num_folds):
        print(f"\n=== Fold {fold_idx + 1}/{num_folds} ===")
        print("seed:100")
        set_seed(0)

        train_files = load_abide_list("train", fold_idx=fold_idx, num_folds=num_folds, site_filter="NYU")
        test_files = load_abide_list("test", fold_idx=fold_idx, num_folds=num_folds, site_filter="NYU")

        train_dataset = GraphDataset(train_files, device, edge_threshold_percentile)

        augmented_dataset = load_augmented_abide_data(
            100,
            f"/yourpath"
            f"hub_0_gamma_1_fold_{fold_idx}_sw_{sw_loss}_ID_drop_{drop_prob}_timestep_{timestep}"
            f"_diff_{diffusion_epoch}_emb_{emb}_hidden_64_vae{vae}_cfg_{guidance_scale}",
            diffusion_epoch,
            edge_threshold_percentile
        )

        combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
        test_dataset = GraphDataset(test_files, device, edge_threshold_percentile)

        train_loader = DataLoader(combined_dataset, batch_size=60, shuffle=True, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False, num_workers=0, pin_memory=False)

        model = GraphSAGE(hidden_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        train_loss, _ = train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, epochs)
        test_metrics = evaluate(model, test_loader)

        wandb.log({
            f'fold_{fold_idx}_train_loss': train_loss,
            f'fold_{fold_idx}_test_acc': test_metrics['accuracy'],
            f'fold_{fold_idx}_test_auc': test_metrics['auc'],
            f'fold_{fold_idx}_test_precision': test_metrics['precision'],
            f'fold_{fold_idx}_test_recall': test_metrics['recall'],
            f'fold_{fold_idx}_test_f1': test_metrics['f1']
        })

        fold_results.append({
            'fold': fold_idx,
            'train_loss': train_loss,
            'test_metrics': test_metrics
        })

        test_accs = [r['test_metrics']['accuracy'] for r in fold_results]
        test_aucs = [r['test_metrics']['auc'] for r in fold_results]
        wandb.log({
            'current_avg_test_acc': float(np.mean(test_accs)),
            'current_avg_test_auc': float(np.mean(test_aucs)),
            'current_std_test_acc': float(np.std(test_accs)),
            'current_std_test_auc': float(np.std(test_aucs))
        })

    test_metrics = {
        'accuracy': float(np.mean([r['test_metrics']['accuracy'] for r in fold_results])),
        'auc': float(np.mean([r['test_metrics']['auc'] for r in fold_results])),
        'precision': float(np.mean([r['test_metrics']['precision'] for r in fold_results])),
        'recall': float(np.mean([r['test_metrics']['recall'] for r in fold_results])),
        'f1': float(np.mean([r['test_metrics']['f1'] for r in fold_results])),
        'sensitivity': float(np.mean([r['test_metrics']['sensitivity'] for r in fold_results])),
        'specificity': float(np.mean([r['test_metrics']['specificity'] for r in fold_results]))
    }
    test_stds = {
        'accuracy': float(np.std([r['test_metrics']['accuracy'] for r in fold_results])),
        'auc': float(np.std([r['test_metrics']['auc'] for r in fold_results])),
        'precision': float(np.std([r['test_metrics']['precision'] for r in fold_results])),
        'recall': float(np.std([r['test_metrics']['recall'] for r in fold_results])),
        'f1': float(np.std([r['test_metrics']['f1'] for r in fold_results])),
        'sensitivity': float(np.std([r['test_metrics']['sensitivity'] for r in fold_results])),
        'specificity': float(np.std([r['test_metrics']['specificity'] for r in fold_results]))
    }

    wandb.log({
        'final_test_acc': test_metrics['accuracy'],
        'final_test_auc': test_metrics['auc'],
        'final_test_precision': test_metrics['precision'],
        'final_test_recall': test_metrics['recall'],
        'final_test_f1': test_metrics['f1'],
        'final_test_sensitivity': test_metrics['sensitivity'],
        'final_test_specificity': test_metrics['specificity'],
        'final_test_acc_std': test_stds['accuracy'],
        'final_test_auc_std': test_stds['auc']
    })

    print("\n=== Final Cross-Validation Results ===")
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']:
        print(f"{metric}: {test_metrics[metric]:.4f} ± {test_stds[metric]:.4f}")

    return fold_results


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="NYU_ID_Diff_GraphSAGE_ABIDE")
    wandb.agent(sweep_id, function=train_and_evaluate_all_seeds_and_folds)
