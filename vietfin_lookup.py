import os
import duckdb
import pandas as pd

class VietfinQueryEngine:
    def __init__(self, data_path="vietfin/data/gold/vietfin_bctc_master.parquet"):
        self.data_path = data_path
        # Nếu chưa có file Parquet, tự động chuyển sang dùng file CSV
        if not os.path.exists(self.data_path):
            self.data_path = "vietfin/data/gold/vietfin_bctc_master.csv"
            
        if not os.path.exists(self.data_path):
            raise FileNotFoundError("Không tìm thấy dữ liệu Gold Layer. Vui lòng chạy pipeline gộp dữ liệu trước!")

    def get_bctc(self, ticker: str, year: int, quarter: int) -> pd.DataFrame:
        """Kịch bản 1: Lấy toàn bộ BCTC của 1 mã trong 1 kỳ cụ thể"""
        query = f"""
            SELECT Item_Name, Value, Period
            FROM '{self.data_path}'
            WHERE UPPER(Ticker) = '{ticker.upper()}'
              AND Year = {year}
              AND Quarter = {quarter}
        """
        return duckdb.query(query).df()

    def search_item_names(self, keyword: str) -> pd.DataFrame:
        """Kịch bản 2: Tìm kiếm chuẩn hóa tên chỉ tiêu tài chính theo từ khóa"""
        query = f"""
            SELECT DISTINCT Item_Name
            FROM '{self.data_path}'
            WHERE LOWER(Item_Name) LIKE '%{keyword.lower()}%'
            ORDER BY Item_Name
        """
        return duckdb.query(query).df()

    def compare_metric_across_stocks(self, item_name_exact: str, year: int, quarter: int, top_n: int = 10) -> pd.DataFrame:
        """Kịch bản 3: So sánh 1 chỉ tiêu chính xác giữa tất cả các mã cổ phiếu"""
        query = f"""
            SELECT Ticker, Item_Name, Value, Period
            FROM '{self.data_path}'
            WHERE Item_Name = '{item_name_exact}'
              AND Year = {year}
              AND Quarter = {quarter}
            ORDER BY Value DESC
            LIMIT {top_n}
        """
        return duckdb.query(query).df()

    def filter_stocks_by_threshold(self, keyword: str, min_value: float, year: int, quarter: int) -> pd.DataFrame:
        """Kịch bản 4: Lọc các mã thỏa mãn chỉ tiêu lớn hơn hoặc bằng ngưỡng thiết lập"""
        query = f"""
            SELECT Ticker, Item_Name, Value, Period
            FROM '{self.data_path}'
            WHERE LOWER(Item_Name) LIKE '%{keyword.lower()}%'
              AND Year = {year}
              AND Quarter = {quarter}
              AND Value >= {min_value}
            ORDER BY Value DESC
        """
        return duckdb.query(query).df()