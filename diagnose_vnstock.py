import os
import logging
from typing import Optional
import pandas as pd
from vnstock import Finance

logger = logging.getLogger(__name__)

class VNStockFinancialsCollector:
    # CHỈ sử dụng các nguồn được vnstock hỗ trợ hiện tại
    VALID_SOURCES = ["KBS", "VCI"]

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            # Thiết lập API key nếu có
            os.environ["VNSTOCK_API_KEY"] = api_key

    def fetch_statement(
        self, 
        ticker: str, 
        statement_type: str, 
        period: str = "year"
    ) -> Optional[pd.DataFrame]:
        """
        Cào BCTC của 1 ticker theo statement_type ('balance_sheet', 'income_statement', 'cash_flow', 'ratio')
        Tự động fallback giữa KBS và VCI khi gặp lỗi hoặc dữ liệu rỗng.
        """
        df = None
        used_source = None

        for source in self.VALID_SOURCES:
            try:
                fin = Finance(symbol=ticker, source=source)
                method = getattr(fin, statement_type, None)
                
                if method is None:
                    logger.error(f"Method '{statement_type}' không tồn tại trên vnstock.Finance")
                    return None

                res = method(period=period)
                
                # Kiểm tra kết quả trả về có dữ liệu thực tế hay không
                if res is not None and isinstance(res, pd.DataFrame) and not res.empty:
                    df = res.copy()
                    used_source = source
                    logger.info(f"{ticker}/{statement_type}/{period}: Lấy dữ liệu thành công từ {source}")
                    break
                else:
                    logger.warning(f"{ticker}/{statement_type}/{period}: Nguồn {source} trả về dữ liệu rỗng")

            except Exception as e:
                logger.warning(f"{ticker}/{statement_type}/{period}: Thất bại khi dùng {source} - Lỗi: {e}")

        if df is None or df.empty:
            logger.error(f"Không thể lấy dữ liệu cho {ticker}/{statement_type}/{period} từ tất cả các nguồn")
            return None

        # Gán thêm metadata nguồn lấy dữ liệu
        df["source_name"] = used_source
        return df

    def _write_bronze(
        self, 
        df: pd.DataFrame, 
        out_path: str, 
        ticker: str, 
        statement_type: str, 
        period: str
    ) -> str:
        """
        Chuẩn hóa kiểu dữ liệu trước khi ghi ra file Parquet tầng Bronze.
        """
        enriched = df.copy()

        # DATA ENGINEERING BEST PRACTICE: 
        # Ép TOÀN BỘ dataframe về kiểu string ở tầng Bronze để triệt tiêu hoàn toàn lỗi 
        # "Mixed types" của PyArrow. Việc xử lý kiểu dữ liệu sẽ đưa về tầng Silver.
        enriched = enriched.astype(str)

        # Tạo đường dẫn thư mục nếu chưa có
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Lưu ra Parquet an toàn
        enriched.to_parquet(out_path, index=False, engine="pyarrow")
        
        return out_path