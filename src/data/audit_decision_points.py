"""
Script kiểm tra tập Decision Points (D_eval) - Giai đoạn 3
Đảm bảo mỗi evaluation unit có đủ: User, Timestamp, Candidate, Target.
"""
from pathlib import Path
import pandas as pd
import sys

def main():
    # Đường dẫn tới file D_eval
    # Nếu chạy từ thư mục gốc của project thì đường dẫn này là chuẩn
    file_path = Path("data/processed/evaluation_events.parquet")
    
    print("=" * 60)
    print(f"AUDIT DECISION POINTS (D_eval)")
    print(f"File: {file_path}")
    print("=" * 60)

    if not file_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(file_path)
        cols = df.columns.tolist()
        
        # Các cột bắt buộc phải có theo định nghĩa Giai đoạn 3
        check_dict = {
            "User (user_id)": "user_id",
            "Timestamp (time_ms)": "time_ms",
            "Observed Candidate (video_id)": "video_id",
            "Target (long_view)": "long_view"
        }
        
        print(f"Tổng số Decision Points (Rows): {len(df):,}\n")
        print("CHECKLIST:")
        
        all_passed = True
        for name, col_name in check_dict.items():
            if col_name in cols:
                print(f"  ✅ PASS | {name}")
            else:
                print(f"  ❌ FAIL | {name} (Missing column: '{col_name}')")
                all_passed = False
                
        print("\n" + "=" * 60)
        if all_passed:
            print("🚀 KẾT LUẬN: D_eval hợp lệ 100%! Đã sẵn sàng cho Giai đoạn 4.")
        else:
            print("⚠️ KẾT LUẬN: D_eval THIẾU thông tin. Vui lòng kiểm tra lại data pipeline!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Lỗi trong quá trình đọc file: {e}")

if __name__ == "__main__":
    main()