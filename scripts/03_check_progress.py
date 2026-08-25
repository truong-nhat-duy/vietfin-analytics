import glob
import pandas as pd

file_paths = glob.glob('data/bronze/financial_statements/**/*.parquet', recursive=True)
if not file_paths:
    print('Chưa tìm thấy dữ liệu trong thư mục Bronze!')
else:
    id_cols = {'ticker', 'statement_type', 'period', 'item', 'item_id', 'item_en', 'source_name', 'ingested_at'}
    records = []
    for fp in file_paths:
        try:
            df = pd.read_parquet(fp)
            if not df.empty:
                t = df['ticker'].iloc[0] if 'ticker' in df.columns else 'N/A'
                tc = [c for c in df.columns if c not in id_cols]
                records.append({'ticker': t, 'num_periods': len(tc)})
        except Exception:
            pass
    if records:
        df_rec = pd.DataFrame(records)
        summary = df_rec.groupby('ticker').agg(so_ky_toi_da=('num_periods', 'max')).reset_index()
        print('='*50)
        print(f'TIẾN ĐỘ CÀO DỮ LIỆU (Đã xong {len(summary)} mã)')
        print('='*50)
        print(summary.to_string(index=False))
