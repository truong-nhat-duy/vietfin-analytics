import os
import duckdb
import pandas as pd

class VietfinQueryEngine:
    def __init__(self, data_path="data/gold/financial_ratios.parquet"):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file: {self.data_path}")

    def get_company_ratios(self, ticker: str) -> pd.DataFrame:
        """Lấy toàn bộ dữ liệu chỉ số tài chính của 1 mã"""
        query = f"""
            SELECT *
            FROM '{self.data_path}'
            WHERE UPPER(ticker) = '{ticker.upper()}'
            ORDER BY period DESC
        """
        return duckdb.query(query).df()

    def get_top_stocks_by_column(self, column_name: str, top_n: int = 10) -> pd.DataFrame:
        """Tra cứu top cổ phiếu theo một chỉ số bất kỳ"""
        query = f"""
            SELECT ticker, period, {column_name}
            FROM '{self.data_path}'
            WHERE {column_name} IS NOT NULL
            ORDER BY {column_name} DESC
            LIMIT {top_n}
        """
        try:
            return duckdb.query(query).df()
        except Exception as e:
            return pd.DataFrame({"Lỗi": [f"Cột '{column_name}' không tồn tại trong dữ liệu."]})

if __name__ == "__main__":
    engine = VietfinQueryEngine()

    print("\n--- 1. KIỂM TRA DỮ LIỆU CỦA FPT ---")
    df_fpt = engine.get_company_ratios(ticker="FPT")
    if not df_fpt.empty:
        print(df_fpt.head(5)) # In ra 5 dòng mới nhất
    else:
        print("Không tìm thấy dữ liệu cho mã FPT.")

    print("\n--- 2. XEM DANH SÁCH CÁC CỘT CHỈ SỐ CÓ SẴN ---")
    # Lấy danh sách cột để bạn biết có thể tra cứu những chỉ số nào
    columns = duckdb.query(f"DESCRIBE SELECT * FROM '{engine.data_path}'").df()
    print(", ".join(columns['column_name'].tolist()))