import pandas as pd
import glob

# Đường dẫn tới thư mục chứa các file parquet của bạn (nhớ sửa lại nếu cần)
file_paths = glob.glob('data/processed/*.parquet')

for f in file_paths:
    print(f"========== Phân tích file: {f} ==========")
    try:
        df = pd.read_parquet(f)
        
        # 1. Báo cáo Missing values (chỉ in ra các cột bị thiếu dữ liệu)
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("✅ Không có missing values nào.")
        else:
            print("❌ Missing values:")
            print(missing)
        
        # 2. Kiểm tra cột thời gian (nếu có)
        if 'time_ms' in df.columns:
            print(f"⏳ Time_ms Min: {df['time_ms'].min()}")
            print(f"⏳ Time_ms Max: {df['time_ms'].max()}")
            
            # 3. Kiểm tra tính thứ tự (Event ordering)
            is_sorted = df['time_ms'].is_monotonic_increasing
            print(f"📈 Các sự kiện đã sắp xếp tăng dần theo thời gian chưa? {'✅ Rồi' if is_sorted else '❌ Chưa'}")
        else:
            print("ℹ️ File này không có cột 'time_ms'.")
            
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
    print("\n")