import requests
from bs4 import BeautifulSoup
import pandas as pd
import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_cafef_ratios(ticker):
    url = f"https://cafef.vn/du-lieu/hose/{ticker.lower()}-tai-chinh.chn"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Bóc tách các chỉ số tài chính từ bảng dữ liệu CafeF
        data = {'ticker': ticker.upper()}
        
        # Tìm bảng chỉ số tài chính (Cấu trúc mẫu của CafeF)
        rows = soup.find_all('tr', id=True)
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                metric_name = cols[0].text.strip()
                latest_val = cols[1].text.strip().replace(',', '')
                
                if 'EPS' in metric_name: data['eps'] = float(latest_val) if latest_val else None
                elif 'P/E' in metric_name: data['pe'] = float(latest_val) if latest_val else None
                elif 'P/B' in metric_name: data['pb'] = float(latest_val) if latest_val else None
                elif 'BVPS' in metric_name: data['bvps'] = float(latest_val) if latest_val else None

        return data
    except Exception as e:
        print(f"Lỗi khi cào mã {ticker}: {e}")
        return None

def sync_to_motherduck(df):
    token = os.getenv("MOTHERDUCK_TOKEN")
    con = duckdb.connect(f"md:vietfin_db?token={token}")
    
    # Tạo bảng staging và merge vào MotherDuck
    con.execute("CREATE TABLE IF NOT EXISTS fact_cafef_ratios AS SELECT * FROM df WHERE 1=0")
    con.execute("INSERT INTO fact_cafef_ratios SELECT * FROM df")
    print("Đã đồng bộ thành công lên MotherDuck Gold Layer!")

# Chạy thử nghiệm
if __name__ == "__main__":
    result = scrape_cafef_ratios("VNM")
    if result:
        df_cafef = pd.DataFrame([result])
        sync_to_motherduck(df_cafef)