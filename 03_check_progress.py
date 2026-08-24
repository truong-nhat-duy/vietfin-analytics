import glob
import os
import pandas as pd

def inspect_bronze_progress(bronze_dir="data/bronze/financial_statements"):
    file_paths = glob.glob(f"{bronze_dir}/**/*.parquet", recursive=True)
    
    if not file_paths:
        print("Chưa tìm thấy dữ liệu Parquet nào trong thư mục Bronze!")
        return

    # Tập hợp các cột định danh không phải là kỳ báo cáo
    id_cols = {"ticker", "statement_type", "period", "item", "item_id", "item_en", "source_name", "ingested_at"}
    
    records = []
    for fp in file_paths:
        try:
            df = pd.read_parquet(fp)
            if df.empty:
                continue
            
            ticker = df["ticker"].iloc[0] if "ticker" in df.columns else "UNKNOWN"
            stmt_type = df["statement_type"].iloc[0] if "statement_type" in df.columns else "N/A"
            period_type = df["period"].iloc[0] if "period" in df.columns else "N/A"
            
            # Cột thời gian là những cột còn lại ngoại trừ id_cols
            time_cols = [c for c in df.columns if c not in id_cols]
            
            records.append({
                "ticker": ticker,
                "statement_type": stmt_type,
                "period_type": period_type,
                "num_periods": len(time_cols),
                "periods_sample": ", ".join(time_cols[:3]) + ("..." if len(time_cols) > 3 else "")
            })
        except Exception:
            continue

    if not records:
        print("Không đọc được dữ liệu hợp lệ.")
        return

    df_rec = pd.DataFrame(records)

    # Thống kê tổng hợp theo từng Mã cổ phiếu
    summary = df_rec.groupby("ticker").agg(
        so_file_bao_cao=("statement_type", "count"),
        so_ky_toi_da=("num_periods", "max"),
        cac_ky_dien_hinh=("periods_sample", lambda x: " | ".join(list(set(x))[:2]))
    ).reset_index()

    # Hiển thị kết quả
    print("=" * 80)
    print(f"BÁO CÁO CHI TIẾT DỮ LIỆU ĐÃ CÀO (Tổng số mã: {len(summary)})")
    print("=" * 80)
    print(summary.to_string(index=False))

    # In con số thống kê tổng quan
    full_8_periods = len(summary[summary["so_ky_toi_da"] >= 8])
    less_than_8 = len(summary[summary["so_ky_toi_da"] < 8])
    
    print("\n" + "-" * 80)
    print(f"-> Số mã đạt đủ tối đa 8 kỳ: {full_8_periods} mã")
    print(f"-> Số mã có ít hơn 8 kỳ (do mới niêm yết/thiếu data): {less_than_8} mã")
    print("-" * 80)

if __name__ == "__main__":
    inspect_bronze_progress()