# PT-Diff

![image](https://github.com/user-attachments/assets/7dfdde82-eb32-4331-b2c2-9386d1bc2eed)


## 📌 Hyperparameter Summary

### 🔹 **Global Configuration**
| Hyperparameter | Value |
|--------------|----|
| `gpu_id` | 2 |
| `vae_epochs` | 200 |
| `cond_epochs` | 1000 |
| `batch_size` | 16 |
| `learning_rate_gae` | 2e-4 |
| `learning_rate_diff` | 1e-3 |
| `guidance_scale` | 11 |
| `cond_drop_prob` | 0.4 |
| `hidden_channels` | 32 |
| `embed_channels` | 8 |

---

### 🔹 **1. Personalized FC Variational Graph Autoencoder**
| Hyperparameter | Value |
|--------------|----|
| `in_channels` | `train_dataset.data_list[0].x.shape[1]` (Node feature dimension) |
| `hidden_channels` | 32 |
| `embed_channels` | 8 |
| `original_feature_dim` | `in_channels` |
| `num_nodes` | 116 |
| `device` | `'cuda'` or `'cpu'` |
| **Edge Decoder** Hidden Dim | 64 |
| **Node Decoder** Hidden Dim | 64 |

---

### 🔹 **2. Diffusion1DCond (Classifier-Free Diffusion Model)**
| Hyperparameter | Value |
|--------------|----|
| `timesteps` | 200 |
| `beta_start` | 1e-4 |
| `beta_end` | 0.02 |
| `device` | `'cuda'` or `'cpu'` |

---

### 🔹 **3. UNet1DStable (1D U-Net with Classifier-Free Guidance)**
| Hyperparameter | Value |
|--------------|----|
| `input_channels` | 1 |
| `base_channels` | 64 |
| `channel_mults` | (1, 2) |
| `use_middle_attn` | True |
| `cond_dim` | 128 |
| `with_time_emb` | True |
| `groups` | 8 |

---

### 🔹 **4. EnhancedConditionEncoder (Condition Embedding for Disease)**
| Hyperparameter | Value |
|--------------|----|
| `emb_dim_disease` | 32 |
| `cond_dim` | 128 |
| `hidden_dim` | 64 |
| `dropout_rate` | 0.1 |

---

### 🔹 **5. Training Configuration**
| Hyperparameter | Value |
|--------------|----|
| `optimizer_gae` | `AdamW(lr=2e-4, betas=(0.9, 0.999))` |
| `scheduler_gae` | `ExponentialLR(gamma=0.998)` |
| `optimizer_diff` | `AdamW(lr=1e-3, betas=(0.9, 0.999))` |
| `scheduler_diff` | `ExponentialLR(gamma=0.998)` |
| `num_neg_samples` | 64000 |
| `latent_dim` | `embed_channels * 2 = 16` |

---
