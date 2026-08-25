from pathlib import Path
import pandas as pd

# Đường dẫn dữ liệu
SILVER_FILE = r"D:\vietfin\data\silver\corporate_master.parquet"
GOLD_DIR = Path(r"D:\vietfin\data\gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

print("--- Bắt đầu bóc tách dữ liệu Gold Layer ---")
df_corp = pd.read_parquet(SILVER_FILE)

# 1. Bảng Hồ sơ doanh nghiệp (dim_company)
df_overview = df_corp[df_corp["dataset"] == "overview"].copy()
company_cols_map = {
    "ticker": "ticker",
    "organ_name_vci": "company_name",
    "icb_name3_vci": "industry",
    "icb_name2_vci": "sector",
    "tax_id_kbs": "tax_id",
    "address_kbs": "address",
    "phone_kbs": "phone",
    "website_kbs": "website",
    "ceo_kbs": "ceo_name",
}
valid_cols = [col for col in company_cols_map.keys() if col in df_overview.columns]
dim_company = (
    df_overview[valid_cols]
    .rename(columns=company_cols_map)
    .drop_duplicates(subset=["ticker"], keep="first")
)

# 2. Bảng Ban lãnh đạo (dim_officers)
df_officers = df_corp[df_corp["dataset"] == "officers"].copy()
dim_officers = df_officers.dropna(how="all", axis=1)

# 3. Bảng Cổ đông lớn (dim_shareholders)
df_shareholders = df_corp[df_corp["dataset"] == "shareholders"].copy()
dim_shareholders = df_shareholders.dropna(how="all", axis=1)

# Ghi file ra tầng Gold
dim_company.to_parquet(GOLD_DIR / "dim_company.parquet", index=False)
dim_officers.to_parquet(GOLD_DIR / "dim_officers.parquet", index=False)
dim_shareholders.to_parquet(GOLD_DIR / "dim_shareholders.parquet", index=False)

print(f"✅ Bóc tách Gold Layer thành công tại: {GOLD_DIR}")
print(f"- dim_company: {len(dim_company)} doanh nghiệp")
print(f"- dim_officers: {len(dim_officers)} bản ghi lãnh đạo")
print(f"- dim_shareholders: {len(dim_shareholders)} bản ghi cổ đông")