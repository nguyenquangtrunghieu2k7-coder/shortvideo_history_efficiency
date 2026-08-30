# -*- coding: utf-8 -*-
"""
Training Loop cho các mô hình Baseline (K=0 và History-only Diagnostic).
Thực hiện huấn luyện, tính toán ROC-AUC và Early Stopping trên tập Validation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Import kiến trúc mô hình đã định nghĩa ở bước trước
from baseline_models import KuaiRandRecommender

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Siêu tham số cố định (Constant Capacity & Protocol)
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 3 # Early stopping patience
NUM_VIDEOS = 7387 # Số lượng video thực tế lớn nhất

# =============================================================================
# DATASET
# =============================================================================
class KuaiRandDataset(Dataset):
    def __init__(self, split_name, k_budget):
        super().__init__()
        print(f"Loading data cho tập {split_name} (K={k_budget})...")
        
        # 1. Load split file (chứa eval_idx và các cột history)
        split_df = pd.read_parquet(PROCESSED_DIR / f"{split_name}.parquet")
        
        # 2. Load evaluation_events (chứa candidate video_id và nhãn long_view)
        events_df = pd.read_parquet(PROCESSED_DIR / "evaluation_events.parquet")
        # Giữ lại các cột cần thiết để merge
        events_df = events_df[['evaluation_id', 'video_id', 'long_view']]
        
        # 3. Kết nối dữ liệu dựa trên eval_idx == evaluation_id
        merged = pd.merge(split_df, events_df, left_on="eval_idx", right_on="evaluation_id", how="inner")
        
        # Trích xuất Tensor cho Candidate và Label
        self.candidate_ids = torch.tensor(merged["video_id"].values, dtype=torch.long)
        self.labels = torch.tensor(merged["long_view"].values, dtype=torch.float32)
        
        # 4. Xử lý mảng history_ids
        if k_budget == 0:
            # Nếu K=0, history là mảng rỗng, ta tạo tensor dummy (tất cả bằng 0)
            self.history_ids = torch.zeros((len(merged), 1), dtype=torch.long)
            self.history_mask = torch.zeros((len(merged), 1), dtype=torch.float32)
        else:
            hist_col = f"history_{k_budget}" if k_budget != "full" else "history_full"
            hist_lists = merged[hist_col].tolist()
            
            # Tính max_len để padding
            max_len = max(len(lst) if lst is not None else 0 for lst in hist_lists)
            max_len = max(max_len, 1) # Ít nhất là 1 để tránh lỗi tensor rỗng
            
            padded_hists = []
            masks = []
            for lst in hist_lists:
                if lst is None or len(lst) == 0:
                    padded_hists.append([0] * max_len)
                    masks.append([0.0] * max_len)
                else:
                    pad_len = max_len - len(lst)
                    padded_hists.append(list(lst) + [0] * pad_len)
                    masks.append([1.0] * len(lst) + [0.0] * pad_len)
                    
            self.history_ids = torch.tensor(padded_hists, dtype=torch.long)
            self.history_mask = torch.tensor(masks, dtype=torch.float32)
            
        print(f"  -> Hoàn tất load {len(self.labels):,} mẫu.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.candidate_ids[idx],
            self.history_ids[idx],
            self.history_mask[idx],
            self.labels[idx]
        )

# =============================================================================
# HÀM HUẤN LUYỆN (TRAINING LOOP)
# =============================================================================
def train_model(mode="video_only", k_budget=0):
    print("=" * 80)
    print(f"BẮT ĐẦU TRAINING: Mode = {mode} | K = {k_budget}")
    print("=" * 80)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")
    
    # 1. Load Data
    train_dataset = KuaiRandDataset("train", k_budget)
    val_dataset = KuaiRandDataset("validation", k_budget)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Khởi tạo Model, Loss, Optimizer
    model = KuaiRandRecommender(num_videos=NUM_VIDEOS, mode=mode).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_auc = 0.0
    patience_counter = 0
    save_path = MODEL_DIR / f"baseline_{mode}_K{k_budget}.pth"
    metrics_path = MODEL_DIR / f"metrics_{mode}_K{k_budget}.csv"
    
    metrics_history = [] # Dùng để lưu log kết quả
    
    # 3. Epoch Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for c_ids, h_ids, h_mask, labels in train_loader:
            c_ids, h_ids, h_mask, labels = c_ids.to(device), h_ids.to(device), h_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            preds = model(c_ids, h_ids, h_mask)
            
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * c_ids.size(0)
            
        train_loss = total_loss / len(train_dataset)
        
        # 4. Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for c_ids, h_ids, h_mask, labels in val_loader:
                c_ids, h_ids, h_mask = c_ids.to(device), h_ids.to(device), h_mask.to(device)
                preds = model(c_ids, h_ids, h_mask)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.numpy())
                
        val_auc = roc_auc_score(val_labels_list, val_preds)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")
        
        # Lưu vào danh sách history
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_auc": val_auc
        })
        
        # 5. Early Stopping & Lưu model đóng băng
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model cải thiện, đã lưu trọng số tại {save_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOPPING] Kích hoạt ở Epoch {epoch+1}. Val AUC tốt nhất: {best_val_auc:.4f}")
                break
                
    # 6. Ghi toàn bộ kết quả ra file CSV
    pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)
    print(f"\n[HOÀN THÀNH] Đã lưu trọng số tại {save_path.name}")
    print(f"[HOÀN THÀNH] Đã lưu file kết quả metrics tại {metrics_path.name}")

if __name__ == "__main__":
    # Huấn luyện Baseline 2: Video-only learned model (K=0)
    train_model(mode="video_only", k_budget=0)
    
    # Huấn luyện Baseline 3: History-only diagnostic (Sử dụng K=Full để chẩn đoán)
    train_model(mode="history_only", k_budget="full")