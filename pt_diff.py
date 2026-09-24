import argparse
import json
import os
import math
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import StratifiedKFold

from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Dataset, Data
from datetime import datetime


EDGE_RATIO = 0.10


def fixed_upper_edge_count(num_nodes):
    possible = num_nodes * (num_nodes - 1) // 2
    return max(1, round(EDGE_RATIO * possible)) if possible else 0


def fc_topk_edges(features):
    num_nodes = features.shape[0]
    row, col = np.triu_indices(num_nodes, k=1)
    scores = np.abs(features[row, col])
    selected = np.argsort(-scores, kind="stable")[:fixed_upper_edge_count(num_nodes)]
    upper = np.stack([row[selected], col[selected]], axis=0)
    reverse = upper[[1, 0]]
    nodes = np.arange(num_nodes)
    loops = np.stack([nodes, nodes], axis=0)
    return np.concatenate([upper, reverse, loops], axis=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)


def load_abide_list(data_type, data_dir, phenotype_csv, fold_idx=0, num_folds=5, seed=100):
    df = pd.read_csv(phenotype_csv)
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip()
    df = df.set_index('FILE_ID', drop=False)

    all_files = sorted(f.name for f in data_dir.glob('*_rois_aal.1D'))
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
        raise ValueError("data_type must be 'train' or 'test'.")
    print(f"Fold {fold_idx+1}/{num_folds}, {data_type} set: {len(selected_files)} files")
    return np.array(selected_files)


class GraphDataset(Dataset):
    def __init__(self, data_filenames, data_dir, phenotype_csv):
        super(GraphDataset, self).__init__()
        self.abide_path = data_dir

        self.df = pd.read_csv(phenotype_csv)
        self.df['FILE_ID'] = self.df['FILE_ID'].astype(str).str.strip()
        self.df = self.df.set_index('FILE_ID', drop=False)
        self.site_to_num = {site: i for i, site in enumerate(self.df['SITE_ID'].unique())}
        self.df['SITE_NUM'] = self.df['SITE_ID'].map(self.site_to_num)

        self.data_filenames = data_filenames
        self.data_list = self._load_data()

    def _load_data(self):
        data_list = []
        for filename in self.data_filenames:
            file_path = self.abide_path / filename
            file_id = filename.replace('_rois_aal.1D', '').strip()

            if file_id not in self.df.index:
                print(f"[WARNING] File {file_id} not in phenotype CSV")
                continue

            roi_signal = np.loadtxt(file_path)
            if np.all(roi_signal != 0):
                corr_matrix = np.corrcoef(roi_signal, rowvar=False)
                corr_matrix = np.arctanh(np.clip(corr_matrix, -0.9999, 0.9999))
                np.fill_diagonal(corr_matrix, 1)

                node_features = corr_matrix
                if not np.isfinite(node_features).all():
                    continue
                edge_index = fc_topk_edges(node_features)

                label = int(self.df.loc[file_id, 'DX_GROUP'] == 1)
                site_num = self.df.loc[file_id, 'SITE_NUM']

                data = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y=torch.tensor(label, dtype=torch.long),
                    site_num=torch.tensor(site_num, dtype=torch.long)
                )
                data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def make_symmetric_with_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    nodes = torch.arange(num_nodes, device=edge_index.device)
    loops = torch.stack([nodes, nodes], dim=0)
    return torch.cat([edge_index, edge_index[[1, 0]], loops], dim=1)


def negative_sampling_upper(data, device, undirected=True):
    adj_b = to_dense_adj(data.edge_index, batch=data.batch).bool().to(device)
    neg_edges = []
    B = adj_b.size(0)

    for b in range(B):
        start = int(data.ptr[b].item())
        end = int(data.ptr[b + 1].item())
        n = end - start
        adj = adj_b[b, :n, :n]

        pos_mask = torch.triu(adj, diagonal=1)
        cand_neg = torch.triu(torch.ones_like(adj, dtype=torch.bool), diagonal=1) & ~pos_mask
        neg_idx = cand_neg.nonzero(as_tuple=False)
        if neg_idx.numel() == 0:
            continue

        k = int(pos_mask.sum().item())
        if k == 0:
            continue

        perm = torch.randperm(neg_idx.size(0), device=device)[:k]
        neg_e_local = neg_idx[perm].t().contiguous()
        neg_e_global = neg_e_local + start

        if undirected:
            neg_e_global = torch.cat([neg_e_global, neg_e_global[[1, 0]]], dim=1)
        neg_edges.append(neg_e_global)

    if len(neg_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(neg_edges, dim=1)


class MLPEdgeDecoder(nn.Module):
    def __init__(self, embed_channels, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3 * embed_channels, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.mish = nn.Mish()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        feat = torch.cat([src + dst, (src - dst).abs(), src * dst], dim=1)
        hidden = self.mish(self.layer_norm1(self.fc1(feat)))
        return self.fc2(hidden).squeeze(1)

    def edge_loss_fn(self, preds, labels):
        return F.binary_cross_entropy_with_logits(preds, labels.float())


class MLPNNodeDecoder(nn.Module):
    def __init__(self, embed_channels, hidden_dim=64, num_nodes=116, original_feature_dim=116):
        super().__init__()
        self.embed_channels = embed_channels
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.original_feature_dim = original_feature_dim
        self.conv1 = SAGEConv(embed_channels, embed_channels)
        self.conv2 = SAGEConv(embed_channels, embed_channels)
        self.fc1 = nn.Linear(embed_channels * num_nodes, hidden_dim * num_nodes)
        self.mish = nn.Mish()
        self.fc2 = nn.Linear(hidden_dim * num_nodes, original_feature_dim * num_nodes)

    def enforce_symmetry(self, z):
        z_hat = (z + z.transpose(1, 2)) / 2.0
        diag = torch.eye(self.num_nodes, dtype=torch.bool, device=z.device)
        diag = diag.unsqueeze(0).expand(z.shape[0], -1, -1)
        return z_hat.masked_fill(diag, 1)

    def forward(self, z, edge_index, batch_size):
        z = self.mish(self.conv1(z, edge_index))
        z = self.mish(self.conv2(z, edge_index))
        z_flat = z.view(batch_size, self.num_nodes * self.embed_channels)
        hidden = self.mish(self.fc1(z_flat))
        x_reconstructed = self.fc2(hidden)
        x_reconstructed = x_reconstructed.view(batch_size, self.num_nodes, self.original_feature_dim)
        x_reconstructed = self.enforce_symmetry(x_reconstructed)
        return x_reconstructed.view(batch_size * self.num_nodes, self.original_feature_dim)

    def node_loss_fn(self, x_reconstructed, x):
        return F.mse_loss(x_reconstructed, x)


class PT_VGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, embed_channels,
                 original_feature_dim, num_nodes=116):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_channels = embed_channels

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.act = nn.Mish()

        self.node_mu = nn.Linear(hidden_channels, embed_channels)
        self.node_logvar = nn.Linear(hidden_channels, embed_channels)

        self.edge_proj = nn.Linear(hidden_channels, embed_channels)
        self.edge_mu = nn.Linear(embed_channels, embed_channels)
        self.edge_logvar = nn.Linear(embed_channels, embed_channels)

        self.edge_decoder = MLPEdgeDecoder(embed_channels, 64)
        self.node_decoder = MLPNNodeDecoder(embed_channels, 64, num_nodes, original_feature_dim)

    def encode_graph(self, x, edge_index_upper):
        edge_full = make_symmetric_with_self_loops(edge_index_upper, x.size(0))
        h = self.act(self.conv1(x, edge_full))
        h = self.act(self.conv2(h, edge_full))
        node_mu, node_lv = self.node_mu(h), self.node_logvar(h)
        z_node = self.reparam(node_mu, node_lv)

        eh = self.act(self.edge_proj(h))
        edge_mu, edge_lv = self.edge_mu(eh), self.edge_logvar(eh)
        z_edge = self.reparam(edge_mu, edge_lv)
        return z_edge, edge_mu, edge_lv, z_node, node_mu, node_lv

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, pos_edge_index, neg_edge_index, original_features, batch):
        z_edge, e_mu, e_lv, z_node, n_mu, n_lv = self.encode_graph(x, pos_edge_index)

        pos_logit = self.edge_decoder(z_edge, pos_edge_index)
        neg_logit = self.edge_decoder(z_edge, neg_edge_index)
        logits = torch.cat([pos_logit, neg_logit])
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)])
        loss_edges = self.edge_decoder.edge_loss_fn(logits, labels)

        keep_edges = []
        candidate_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        # Use sampled BCE candidates, but fix the output count relative to all pairs.
        B = int(batch.max().item()) + 1
        for g in range(B):
            nodes = (batch == g).nonzero(as_tuple=False).flatten()
            mask = (batch[candidate_edges[0]] == g) & (batch[candidate_edges[1]] == g)
            graph_edges = candidate_edges[:, mask]
            candidate_logits = logits[mask]
            k = fixed_upper_edge_count(nodes.numel())
            if graph_edges.shape[1] < k:
                raise ValueError(f"graph {g}: {graph_edges.shape[1]} candidates cannot supply {k} edges")
            top_idx = torch.topk(candidate_logits, k=k).indices
            keep_edges.append(graph_edges[:, top_idx])

        rec_upper = torch.cat(keep_edges, dim=1)
        edge_full = make_symmetric_with_self_loops(rec_upper, x.size(0))

        x_rec = self.node_decoder(z_node, edge_full, B)
        loss_nodes = self.node_decoder.node_loss_fn(x_rec, original_features)

        kl = -0.5 * torch.mean(1 + e_lv - e_mu.pow(2) - e_lv.exp()) + -0.5 * torch.mean(
            1 + n_lv - n_mu.pow(2) - n_lv.exp()
        )
        kl *= 0.5

        total = loss_edges + loss_nodes + 0.1 * kl
        return total, logits, labels, x_rec

    def graph_to_latent(self, x, edge_index_upper):
        z_edge, _, _, z_node, _, _ = self.encode_graph(x, edge_index_upper)
        return torch.cat([z_edge, z_node], dim=1)

    def latent_to_graph(self, z_latent, batch_size, num_nodes=116):
        z_edge, z_node = torch.chunk(z_latent, chunks=2, dim=1)

        i, j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=z_latent.device)
        edge_upper = torch.stack([i, j], dim=0)

        logits = self.edge_decoder(z_edge, edge_upper)
        K = fixed_upper_edge_count(num_nodes)
        topk_idx = torch.topk(logits, K).indices
        rec_upper = edge_upper[:, topk_idx]
        selected_edges = make_symmetric_with_self_loops(rec_upper, num_nodes)

        x_reconstructed = self.node_decoder(z_node, selected_edges, batch_size)
        return x_reconstructed, selected_edges

    def latent_to_graph_batch(self, z_batch: torch.Tensor, num_nodes=116):
        B, N, _ = z_batch.shape
        if N != num_nodes:
            raise ValueError("Latent node count does not match num_nodes.")
        out_x, out_e = [], []
        for b in range(B):
            x_r, e_r = self.latent_to_graph(z_batch[b], 1, num_nodes)
            out_x.append(x_r)
            out_e.append(e_r)
        return torch.stack(out_x), out_e


def train_gae(model, optimizer, scheduler, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        pos_edge_index_full = data.edge_index
        mask_upper = pos_edge_index_full[0] < pos_edge_index_full[1]
        pos_edge_index = pos_edge_index_full[:, mask_upper]
        neg_edge_index = negative_sampling_upper(data, device=device, undirected=False)
        original_features = data.x

        loss, _, _, _ = model(data.x, pos_edge_index, neg_edge_index, original_features, data.batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step()
    return total_loss / len(train_loader)


class EnhancedConditionEncoder(nn.Module):
    def __init__(self, emb_dim_disease=32, cond_dim=128, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.disease_emb = nn.Embedding(2, emb_dim_disease)
        self.fc1 = nn.Linear(emb_dim_disease, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, cond_dim)

    def forward(self, disease: torch.Tensor) -> torch.Tensor:
        if disease.dim() == 2 and disease.size(-1) == 1:
            disease = disease.squeeze(-1)
        disease = disease.long()

        d_emb = self.disease_emb(disease)
        x = self.fc1(d_emb)
        x = self.layer_norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        cond = self.fc2(x)
        return cond


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class NodeTransformerEps(nn.Module):
    def __init__(self, latent_dim=64, depth=6, heads=8, cond_dim=128, num_nodes=116, use_cross_attn=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes

        self.node_id_emb = nn.Embedding(num_nodes, latent_dim)

        self.time_pos = SinusoidalPosEmb(latent_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4), nn.Mish(), nn.Linear(latent_dim * 4, latent_dim)
        )

        self.use_cross = use_cross_attn
        self.cond_proj = nn.Linear(cond_dim, latent_dim)

        blocks = []
        for _ in range(depth):
            self_attn = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
            cross = nn.MultiheadAttention(latent_dim, heads, batch_first=True) if use_cross_attn else None
            ffn = FFN(latent_dim)
            blocks.append(nn.ModuleList([self_attn, ffn, cross]))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(latent_dim)
        self.to_eps = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, t, cond=None):
        B, N, D = x.shape

        x = x + self.node_id_emb.weight[:N].unsqueeze(0)
        t_emb = self.time_mlp(self.time_pos(t))
        x = x + t_emb.unsqueeze(1)

        c = None
        if (cond is not None) and self.use_cross:
            c = self.cond_proj(cond).unsqueeze(1)

        for self_attn, ffn, cross in self.blocks:
            x2, _ = self_attn(x, x, x)
            x = x + x2

            if cross is not None and c is not None:
                x3, _ = cross(x, c, c)
                x = x + x3

            x = x + ffn(self.norm(x))

        return self.to_eps(self.norm(x))


class DiffusionNodes(nn.Module):
    def __init__(self, timesteps=1000, device='cuda'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.betas, self.alphas, self.alphas_cumprod = self._cosine_schedule(timesteps, device)

    @staticmethod
    def _cosine_schedule(T, device, s=0.008):
        t = np.linspace(0, T, T + 1)
        f = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
        abar = f / f[0]
        betas = 1 - (abar[1:] / abar[:-1])
        betas = np.clip(betas, 1e-6, 0.999).astype(np.float32)
        betas = torch.from_numpy(betas).to(device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        return betas, alphas, abar

    def q_sample(self, x0, t):
        abar = self.alphas_cumprod[t].view(-1, 1, 1)
        noise = torch.randn_like(x0)
        xt = abar.sqrt() * x0 + (1.0 - abar).sqrt() * noise
        return xt, noise

    def p_losses_v_weighted(self, eps_model, x0, t, cond=None, gamma: float = 5.0):
        xt, noise = self.q_sample(x0, t)
        abar = self.alphas_cumprod[t].view(-1, 1, 1)
        v_tgt = abar.sqrt() * noise - (1.0 - abar).sqrt() * x0
        v_pred = eps_model(xt, t, cond=cond)
        snr = abar / (1.0 - abar)
        w = torch.minimum(snr, torch.full_like(snr, gamma))
        return (w * (v_pred - v_tgt) ** 2).mean()


@torch.no_grad()
def sample_with_ddim_cfg_nodes_v(eps_model, diffusion, cond, B, N, D,
                                 steps=200, eta=0.0, guidance=7.5, device='cuda'):
    xt = torch.randn(B, N, D, device=device)
    T = diffusion.timesteps
    idx = torch.linspace(0, T - 1, steps, device=device).long()
    abar = diffusion.alphas_cumprod[idx]
    abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]])

    for i in reversed(range(steps)):
        ti = idx[i]
        abar_t, abar_tm1 = abar[i], abar_prev[i]
        t_vec = torch.full((B,), int(ti.item()), device=device, dtype=torch.long)

        v0 = eps_model(xt, t_vec, cond=None)
        if cond is not None:
            v1 = eps_model(xt, t_vec, cond=cond)
            v = v0 + guidance * (v1 - v0)
        else:
            v = v0

        sqrt_abar = abar_t.sqrt().view(1, 1, 1)
        sqrt_1mabar = (1.0 - abar_t).sqrt().view(1, 1, 1)

        x0 = sqrt_abar * xt - sqrt_1mabar * v
        eps = sqrt_1mabar * xt + sqrt_abar * v

        sigma_t = eta * torch.sqrt((1.0 - abar_tm1) / (1.0 - abar_t) * (1.0 - abar_t / abar_tm1))
        sigma_t = sigma_t.clamp(min=0.0).view(1, 1, 1)
        dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_tm1 - sigma_t ** 2, min=0.0)).view(1, 1, 1)

        xt = abar_tm1.sqrt().view(1, 1, 1) * x0 + dir_coeff * eps
        if i > 0 and eta > 0.0:
            xt = xt + sigma_t * torch.randn_like(xt)
    return xt


def compute_latent_stats(model, loader, device, num_nodes, dim):
    model.eval()
    s = torch.zeros(dim, device=device)
    ss = torch.zeros(dim, device=device)
    n = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            B = int(data.batch.max().item() + 1)
            ei = data.edge_index
            upper = ei[:, ei[0] < ei[1]]
            z0 = model.graph_to_latent(data.x, upper)
            z0 = z0.view(B, num_nodes, dim)
            zf = z0.reshape(-1, dim)
            s += zf.sum(0)
            ss += (zf * zf).sum(0)
            n += zf.size(0)
    mean = s / max(n, 1)
    var = ss / max(n, 1) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std


def get_graph_labels_from_batch(batch: Data) -> torch.Tensor:
    B = int(batch.batch.max().item()) + 1
    if not hasattr(batch, "y") or batch.y is None:
        return torch.zeros(B, dtype=torch.long, device=batch.x.device)

    y = batch.y
    if y.dim() == 0:
        return y.view(1).repeat(B)
    if y.dim() == 2 and y.size(0) == B and y.size(1) == 1:
        return y.squeeze(1)
    if y.size(0) == B:
        return y.view(-1)
    if y.size(0) == batch.x.size(0):
        idx = batch.ptr[:-1].to(y.device)
        return y.index_select(0, idx).view(-1)
    raise RuntimeError(
        f"Unexpected batch.y shape: {tuple(y.shape)}; expected (B,), (B,1), scalar, or (total_nodes,)."
    )


def main(data_dir, phenotype_csv, output_dir, config):
    run_dir = output_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_cfg{config['guidance_scale']:g}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(dict(config, data_dir=str(data_dir),
        phenotype_csv=str(phenotype_csv), edge_ratio=EDGE_RATIO), indent=2) + "\n")

    def log_metrics(values):
        with (run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(values, fold=fold_idx), allow_nan=False) + "\n")

    device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    seeds = [config["seed"]]
    num_folds = config["num_folds"]

    for seed in seeds:
        set_seed(seed)
        for fold_idx in range(num_folds):
            if fold_idx not in config["folds"]:
                continue
            print(f"\n=== Training with seed={seed}, fold={fold_idx+1}/{num_folds} ===")
            result_dir = run_dir / f"fold_{fold_idx}"
            os.makedirs(result_dir, exist_ok=True)
            samples_dir = os.path.join(result_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)

            train_files = load_abide_list("train", data_dir, phenotype_csv, fold_idx=fold_idx, num_folds=num_folds, seed=seed)

            train_dataset = GraphDataset(train_files, data_dir, phenotype_csv)
            if config["smoke"]:
                train_dataset.data_list = train_dataset.data_list[:128]
            if len(train_dataset) < config["batch_size"]:
                raise ValueError("Training set is smaller than batch_size after filtering.")
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(seed),
                drop_last=True,
                pin_memory=False
            )

            in_channels = train_dataset.data_list[0].x.shape[1]
            hidden_channels = config["hidden_channels"]
            embed_channels = config["embed_channels"]
            num_nodes = config["num_nodes"]
            if in_channels != num_nodes:
                raise ValueError("Data node count does not match num_nodes.")

            model = PT_VGAE(in_channels, hidden_channels, embed_channels, in_channels, num_nodes).to(device)
            initialize_weights(model)
            print("Initialized PT_VGAE")

            optimizer_gae = optim.AdamW(
                model.parameters(),
                lr=config["learning_rate_gae"],
                weight_decay=1e-04,
                betas=(0.9, 0.999)
            )
            num_epochs_gae = config['vae_epochs']
            from torch.optim.lr_scheduler import OneCycleLR
            steps_per_epoch_gae = len(train_loader)
            total_steps_gae = config['vae_epochs'] * steps_per_epoch_gae

            base_lr = float(config["learning_rate_gae"])
            max_lr = base_lr * 5

            scheduler_gae = OneCycleLR(
                optimizer_gae,
                max_lr=max_lr,
                total_steps=total_steps_gae,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=1000.0
            )

            print(f"Starting GAE training for {num_epochs_gae} epochs")
            for epoch in range(1, num_epochs_gae + 1):
                train_loss = train_gae(model, optimizer_gae, scheduler_gae, train_loader, device)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{num_epochs_gae}, GAE train loss: {train_loss:.4f}")
                log_metrics({"phase": "GAE_train", "vae_epoch": epoch, "train_loss": train_loss, "seed": seed})

            D = embed_channels * 2
            latent_mean, latent_std = compute_latent_stats(model, train_loader, device, num_nodes, D)
            latent_mean_b = latent_mean.view(1, 1, D)
            latent_std_b = latent_std.view(1, 1, D)
            print("[Latent stats] mean|std:", float(latent_mean.abs().mean()), float(latent_std.mean()))

            cond_enc = EnhancedConditionEncoder().to(device)
            eps_model = NodeTransformerEps(
                latent_dim=embed_channels * 2,
                depth=6,
                heads=8,
                cond_dim=128,
                num_nodes=num_nodes,
                use_cross_attn=True
            ).to(device)
            initialize_weights(cond_enc)
            initialize_weights(eps_model)

            diffusion_cond = DiffusionNodes(timesteps=config['timesteps'], device=device)
            optimizer_cond = optim.AdamW(
                list(eps_model.parameters()) + list(cond_enc.parameters()),
                weight_decay=1e-04,
                lr=config["learning_rate_diff"],
                betas=(0.9, 0.999)
            )
            num_epochs_diff = config["cond_epochs"]
            steps_per_epoch_diff = len(train_loader)
            total_steps_diff = num_epochs_diff * steps_per_epoch_diff

            base_lr_diff = float(config["learning_rate_diff"])
            max_lr_diff = base_lr_diff * 5.0

            scheduler_cond = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_cond,
                max_lr=max_lr_diff,
                total_steps=total_steps_diff,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=1000.0
            )
            print(f"Starting Transformer Diffusion training for {num_epochs_diff} epochs with OneCycleLR")

            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            cond_drop_prob = config["cond_drop_prob"]

            steps_per_epoch = len(train_loader)
            total_diff_steps = num_epochs_diff * steps_per_epoch

            pbar_all = tqdm(
                total=total_diff_steps,
                desc=f"Diff total ({num_epochs_diff}e x {steps_per_epoch}b)",
                dynamic_ncols=True,
                position=0,
                leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            global_step = 0

            for epoch in range(1, num_epochs_diff + 1):
                eps_model.train()
                cond_enc.train()
                total_loss = 0.0

                pbar_epoch = tqdm(
                    train_loader,
                    desc=f"Diff ep {epoch}/{num_epochs_diff}",
                    dynamic_ncols=True,
                    position=1,
                    leave=False
                )

                running = None
                for step, data in enumerate(pbar_epoch, start=1):
                    data = data.to(device)
                    B = int(data.batch.max().item() + 1)
                    with torch.no_grad():
                        ei = data.edge_index
                        upper = ei[:, ei[0] < ei[1]]
                        z0 = model.graph_to_latent(data.x, upper).view(B, num_nodes, embed_channels * 2)

                    disease = get_graph_labels_from_batch(data).to(device).long()
                    cond_vec = None if (random.random() < cond_drop_prob) else cond_enc(disease)

                    t = torch.randint(0, diffusion_cond.timesteps, (B,), device=device).long()
                    z0n = (z0 - latent_mean_b) / latent_std_b
                    loss_diff = diffusion_cond.p_losses_v_weighted(eps_model, z0n, t, cond=cond_vec, gamma=1)

                    optimizer_cond.zero_grad()
                    loss_diff.backward()
                    torch.nn.utils.clip_grad_norm_(eps_model.parameters(), 1)
                    optimizer_cond.step()
                    scheduler_cond.step()

                    total_loss += loss_diff.item()
                    running = loss_diff.item() if running is None else 0.9 * running + 0.1 * loss_diff.item()
                    lr_now = scheduler_cond.get_last_lr()[0]

                    pbar_epoch.set_postfix(loss=f"{running:.4f}", lr=f"{lr_now:.2e}")
                    pbar_all.update(1)

                    global_step += 1
                    if global_step % 10 == 0:
                        log_metrics({"diff/global_step": global_step, "diff/lr": lr_now})

                diff_loss = total_loss / steps_per_epoch
                log_metrics(
                    {"diff/global_step": global_step, "diff/loss": diff_loss, "diff/epoch": epoch, "seed": seed}
                )

                if epoch % 10 == 0:
                    print(f"[TransDiff] seed={seed}, epoch={epoch}/{num_epochs_diff}, loss={diff_loss:.4f}")

                if epoch % config['cond_epochs'] == 0:
                    eps_model.eval()
                    cond_enc.eval()
                    model_save_dir = os.path.join(result_dir, "models")
                    os.makedirs(model_save_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f"gae_seed_{seed}_epoch_{epoch}.pth"))
                    torch.save(cond_enc.state_dict(), os.path.join(model_save_dir, f"cond_enc_seed_{seed}_epoch_{epoch}.pth"))
                    torch.save(eps_model.state_dict(), os.path.join(model_save_dir, f"eps_model_seed_{seed}_epoch_{epoch}.pth"))
                    print(f"Saved models at epoch {epoch}")

                    real_labels = [data.y.item() for data in train_dataset.data_list]
                    class_counts = Counter(real_labels)
                    desired_counts = {cls: class_counts.get(1 - cls, 0) for cls in class_counts}
                    if config["smoke"]:
                        desired_counts = {cls: min(2, count) for cls, count in desired_counts.items()}
                    generated_counts = {cls: 0 for cls in desired_counts.keys()}

                    all_node_features, all_edge_indices, all_labels = [], [], []
                    sample_bs = config["sample_batch_size"]
                    guide = config["guidance_scale"]
                    ddim_steps = int(os.environ.get("DDIM_STEPS", config['timesteps'] / 5))

                    for cls in [0, 1]:
                        need = desired_counts[cls] - generated_counts[cls]
                        if need <= 0:
                            continue

                        gen_pbar = tqdm(total=need, desc=f"Sampling cls={cls}", dynamic_ncols=True, leave=False)
                        while need > 0:
                            bs = min(sample_bs, need)
                            lab = torch.full((bs,), cls, device=device)
                            with torch.no_grad():
                                cond_vec = cond_enc(lab)

                            z_batch_norm = sample_with_ddim_cfg_nodes_v(
                                eps_model,
                                diffusion_cond,
                                cond=cond_vec,
                                B=bs,
                                N=num_nodes,
                                D=embed_channels * 2,
                                steps=ddim_steps,
                                eta=float(os.environ.get("DDIM_ETA", 0.15)),
                                guidance=guide,
                                device=device
                            )
                            z_batch = z_batch_norm * latent_std_b + latent_mean_b

                            x_rec, e_rec = model.latent_to_graph_batch(z_batch, num_nodes=num_nodes)

                            for b in range(bs):
                                x2d = x_rec[b].detach().cpu().numpy()
                                x2d = np.nan_to_num(x2d, nan=0.0, posinf=0.0, neginf=0.0)
                                all_node_features.append(x2d)
                                all_labels.append(int(cls))

                                edge_to_save = e_rec[b].detach().cpu().numpy()
                                all_edge_indices.append(edge_to_save)

                                plt.figure(figsize=(6, 5))
                                sns.heatmap(x2d, cmap='jet')
                                plt.title(f"DDIM seed={seed}, epoch={epoch}, cls={cls}")
                                plt.tight_layout()
                                out_png = os.path.join(samples_dir, f"sub_epoch{epoch}_cls{cls}_idx{len(all_labels)}.png")
                                plt.savefig(out_png)
                                plt.close()

                            generated_counts[cls] += bs
                            need -= bs
                            gen_pbar.update(bs)
                        gen_pbar.close()

                    data_save_dir = os.path.join(result_dir, "generated_data")
                    os.makedirs(data_save_dir, exist_ok=True)
                    sample_save_path = os.path.join(data_save_dir, f"all_matched_samples_seed_{seed}_epoch_{epoch}.npz")

                    node_features_np = np.stack(all_node_features, axis=0)
                    edge_indices_np = np.stack(all_edge_indices, axis=0)
                    labels_np = np.asarray(all_labels, dtype=np.int64)

                    np.savez_compressed(
                        sample_save_path,
                        node_features=node_features_np,
                        edge_indices=edge_indices_np,
                        labels=labels_np
                    )
                    print(f"Saved all matched samples at {sample_save_path}")
            pbar_all.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--phenotype-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--gpu-id", type=int, default=0, help="Logical GPU index; prefer CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--vae-epochs", type=int, default=100)
    parser.add_argument("--cond-epochs", type=int, default=100)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--guidance-scales", nargs="+", type=float, default=[9.5, 10.0])
    parser.add_argument("--smoke", action="store_true", help="One fold, 128 training subjects, one epoch per stage, at most 4 synthetic samples.")
    args = parser.parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.phenotype_csv = args.phenotype_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.data_dir.is_dir() or not args.phenotype_csv.is_file():
        parser.error("--data-dir must be a directory and --phenotype-csv must be a file")

    if args.num_folds < 2 or min(args.vae_epochs, args.cond_epochs) < 1:
        parser.error("num-folds must be >=2 and epochs positive")
    folds = args.folds if args.folds is not None else list(range(args.num_folds))
    if any(f < 0 or f >= args.num_folds for f in folds):
        parser.error("fold indices must be in [0, num-folds)")
    if args.smoke:
        folds = folds[:1]
    config = dict(seed=args.seed, num_nodes=116, num_folds=args.num_folds, folds=folds,
        gpu_id=args.gpu_id, vae_epochs=1 if args.smoke else args.vae_epochs,
        cond_epochs=1 if args.smoke else args.cond_epochs, batch_size=64, sample_batch_size=64,
        learning_rate_gae=1e-4, learning_rate_diff=1e-4, cond_drop_prob=.1,
        hidden_channels=64, embed_channels=32, timesteps=100, smoke=args.smoke)
    scales = args.guidance_scales[:1] if args.smoke else args.guidance_scales
    for scale in scales:
        main(args.data_dir, args.phenotype_csv, args.output_dir, dict(config, guidance_scale=scale))
