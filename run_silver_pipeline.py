import ast
import glob
from pathlib import Path
import pandas as pd


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột kiểu object về dạng chuỗi thuần để tránh lỗi PyArrow hỗn hợp kiểu dữ liệu."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(lambda x: None if pd.isna(x) else str(x))
    return out


def process_silver_layer(bronze_paths, output_path, fill_na_val=None):
    df_list = []
    
    print(f"--- Bắt đầu xử lý cho file đầu ra: {output_path} ---")
    for path in bronze_paths:
        search_pattern = f"{path}/**/*.parquet"
        files = glob.glob(search_pattern, recursive=True)
        print(f"Đang tìm trong '{path}': phát hiện {len(files)} file .parquet")
        
        for f in files:
            try:
                df_list.append(pd.read_parquet(f))
            except Exception as e:
                print(f"⚠️ Lỗi đọc file {f}: {e}")
            
    if not df_list:
        print(f"❌ Không tìm thấy dữ liệu nào cho {output_path}.\n")
        return

    # 1. Gộp tất cả các DataFrame
    df = pd.concat(df_list, ignore_index=True)
    print(f"-> Tổng số dòng sau khi gộp: {len(df)}")
    
    # 2. Xóa các dòng trùng lặp toàn bộ
    df = df.drop_duplicates()
    
    # Lọc mã chứng quyền/mã rác (chỉ giữ mã cổ phiếu chuẩn <= 3 ký tự)
    if 'ticker' in df.columns:
        df = df[df['ticker'].astype(str).str.len() <= 3]
        
    # 3. Phục hồi các cột chứa dict/list bị stringified ở Bronze layer
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('{') else x
            )

    # 4. Xử lý giá trị Null / NaN THÔNG MINH (chỉ lấp đầy 0 cho các cột số)
    if fill_na_val is not None:
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(fill_na_val)
    else:
        df = df.dropna(how='all')

    # 5. Làm sạch kiểu dữ liệu cột object trước khi lưu file Parquet
    df = sanitize_for_parquet(df)

    # Tạo thư mục đầu ra nếu chưa tồn tại và ghi file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_parquet(output_file, index=False)
    except Exception:
        # Fallback an toàn nếu vẫn xảy ra xung đột kiểu dữ liệu
        df.astype(str).to_parquet(output_file, index=False)

    print(f"✅ ĐÃ HOÀN THÀNH: Lưu tại '{output_path}' với {len(df)} dòng dữ liệu chuẩn.\n")


if __name__ == "__main__":
    SILVER_BASE = "D:/vietfin/data/silver"

    corporate_paths = [
        r"D:\vietfin\data\bronze\corporate\officers",
        r"D:\vietfin\data\bronze\corporate\overview",
        r"D:\vietfin\data\bronze\corporate\price_history",
        r"D:\vietfin\data\bronze\corporate\ratio_summary",
        r"D:\vietfin\data\bronze\corporate\shareholders"
    ]

    financial_paths = [
        r"D:\vietfin\data\bronze\financial_statements\balance_sheet\quarter",
        r"D:\vietfin\data\bronze\financial_statements\balance_sheet\year",
        r"D:\vietfin\data\bronze\financial_statements\cash_flow\quarter",
        r"D:\vietfin\data\bronze\financial_statements\cash_flow\year",
        r"D:\vietfin\data\bronze\financial_statements\income_statement\quarter",
        r"D:\vietfin\data\bronze\financial_statements\income_statement\year",
        r"D:\vietfin\data\bronze\financial_statements\ratio\quarter",
        r"D:\vietfin\data\bronze\financial_statements\ratio\year"
    ]

    # Run Corporate Master
    process_silver_layer(
        bronze_paths=corporate_paths, 
        output_path=f"{SILVER_BASE}/corporate_master.parquet"
    )

    # Run Financials Master
    process_silver_layer(
        bronze_paths=financial_paths, 
        output_path=f"{SILVER_BASE}/financials_master.parquet",
        fill_na_val=0
    )