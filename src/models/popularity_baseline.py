# -*- coding: utf-8 -*-
"""
Xây dựng Popularity Baseline (Phase 8)
Tính toán độ phổ biến từ Standard Interaction Logs thuộc Training Period.
"""

from pathlib import Path
import pandas as pd
import sys

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw" / "KuaiRand-1K" / "data"

TRAIN_SPLIT_FILE = PROCESSED_DIR / "train.parquet"
STANDARD_FILES = [
    RAW_DIR / "log_standard_4_08_to_4_21_1k.csv",
    RAW_DIR / "log_standard_4_22_to_5_08_1k.csv",
]

# File output để save baseline
POPULARITY_OUT_FILE = PROCESSED_DIR / "popularity_baseline.csv"
LOG_OUT_FILE = PROCESSED_DIR / "build_popularity_log.txt"

# Chuyển hướng log ra file để không bị trôi terminal
sys.stdout = open(LOG_OUT_FILE, 'w', encoding='utf-8')

def main():
    print("=" * 80)
    print("PHASE 8: BUILD POPULARITY BASELINE")
    print("=" * 80)
    
    # 1. Xác định ranh giới thời gian của Training Period
    print("\n1. Xác định mốc thời gian Training Period...")
    if not TRAIN_SPLIT_FILE.exists():
        print(f"[LỖI] Không tìm thấy file {TRAIN_SPLIT_FILE}")
        return
        
    df_train_eval = pd.read_parquet(TRAIN_SPLIT_FILE)
    # Lấy mốc thời gian lớn nhất trong tập train làm ranh giới
    train_max_time = df_train_eval['time_ms'].max()
    print(f"   -> Train Split Max time_ms: {train_max_time}")
    
    # 2. Đọc và lọc raw standard logs
    print("\n2. Đọc và lọc Raw Standard Interaction Logs...")
    raw_dfs = []
    for file_path in STANDARD_FILES:
        if file_path.exists():
            print(f"   -> Đang đọc: {file_path.name}")
            df_chunk = pd.read_csv(file_path, usecols=['video_id', 'time_ms', 'long_view'])
            raw_dfs.append(df_chunk)
        else:
            print(f"   [CẢNH BÁO] Không tìm thấy {file_path.name}")
            
    df_raw = pd.concat(raw_dfs, ignore_index=True)
    print(f"   -> Tổng số raw interactions ban đầu: {len(df_raw):,}")
    
    # LỌC LEAKAGE: Chỉ giữ lại các interaction <= train_max_time
    df_train_raw = df_raw[df_raw['time_ms'] <= train_max_time].copy()
    print(f"   -> Tổng số interactions trong Training Period: {len(df_train_raw):,}")
    print(f"   -> Số interactions bị loại bỏ (tương lai): {len(df_raw) - len(df_train_raw):,}")
    
    # 3. Tính toán độ phổ biến (Popularity)
    print("\n3. Tính toán độ phổ biến (Popularity) theo video_id...")
    
    # Đếm số lần hiển thị (frequency) và số lượng long_view
    popularity_df = df_train_raw.groupby('video_id').agg(
        impression_count=('time_ms', 'count'),
        long_view_count=('long_view', 'sum')
    ).reset_index()
    
    # Tính tỷ lệ long_view rate (có thể cộng thêm epsilon smoothing nếu cần, ở đây để raw)
    popularity_df['long_view_rate'] = popularity_df['long_view_count'] / popularity_df['impression_count']
    
    # Sắp xếp theo độ phổ biến (impression_count)
    popularity_df = popularity_df.sort_values(by='impression_count', ascending=False)
    
    print(f"   -> Số lượng video độc nhất có lịch sử trong Training: {len(popularity_df):,}")
    print("\nTOP 5 VIDEO PHỔ BIẾN NHẤT:")
    print(popularity_df.head(5).to_string())
    
    # 4. Đóng băng và lưu kết quả
    print("\n4. Lưu và đóng băng Popularity Baseline...")
    popularity_df.to_csv(POPULARITY_OUT_FILE, index=False)
    print(f"   -> Đã lưu tại: {POPULARITY_OUT_FILE}")
    print("\n[HOÀN THÀNH] Popularity Baseline đã sẵn sàng để tra cứu (lookup) ở giai đoạn đánh giá!")

if __name__ == "__main__":
    main()