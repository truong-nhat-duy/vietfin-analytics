from pathlib import Path
import pandas as pd

SILVER_DIR = Path(r"D:\vietfin\data\silver")
GOLD_DIR = Path(r"D:\vietfin\data\gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

print("--- 1. Đang xử lý Fact từ corporate_master.parquet ---")
df_corp = pd.read_parquet(SILVER_DIR / "corporate_master.parquet")

# 1.1 Fact Lịch sử giá (fact_daily_prices)
df_prices = df_corp[df_corp["dataset"] == "price_history"].copy()
df_prices = df_prices.dropna(how="all", axis=1)
df_prices.to_parquet(GOLD_DIR / "fact_daily_prices.parquet", index=False)
print(f"✅ fact_daily_prices: {len(df_prices)} dòng")

# 1.2 Fact Chuỗi tỷ số tài chính (fact_ratio_summary)
df_ratios = df_corp[df_corp["dataset"] == "ratio_summary"].copy()
df_ratios = df_ratios.dropna(how="all", axis=1)
df_ratios.to_parquet(GOLD_DIR / "fact_ratio_summary.parquet", index=False)
print(f"✅ fact_ratio_summary: {len(df_ratios)} dòng")

print("\n--- 2. Đang xử lý Fact từ financials_master.parquet ---")
# 1.3 Fact Báo cáo tài chính tổng hợp (fact_financials)
df_fin = pd.read_parquet(SILVER_DIR / "financials_master.parquet")
df_fin.to_parquet(GOLD_DIR / "fact_financials.parquet", index=False)
print(f"✅ fact_financials: {len(df_fin)} dòng")

print(f"\n🎉 Hoàn thành xuất toàn bộ bảng Fact ra: {GOLD_DIR}")