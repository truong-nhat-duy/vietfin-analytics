# config.py
"""
File cấu hình hệ thống VietFin Intelligence:
- Từ điển tọa độ địa lý đồng bộ VN_COORDS
- Hàm helper trích xuất tọa độ & phân vùng địa lý từ địa chỉ doanh nghiệp
"""

import numpy as np
import pandas as pd

# Từ điển tọa độ các Tỉnh / Thành phố tại Việt Nam
VN_COORDS = {
    "hồ chí minh": (10.762622, 106.660172),
    "hcm": (10.762622, 106.660172),
    "tp.hcm": (10.762622, 106.660172),
    "tp hồ chí minh": (10.762622, 106.660172),
    "hà nội": (21.028511, 105.804817),
    "ha noi": (21.028511, 105.804817),
    "tp hà nội": (21.028511, 105.804817),
    "đà nẵng": (16.054407, 108.202167),
    "hải phòng": (20.844912, 106.688084),
    "cần thơ": (10.045162, 105.746853),
    "bình dương": (11.229415, 106.626359),
    "đồng nai": (10.946458, 106.824248),
    "bà rịa - vũng tàu": (10.497557, 107.168535),
    "vũng tàu": (10.497557, 107.168535),
    "long an": (10.536400, 106.406700),
    "bắc ninh": (21.186100, 106.076300),
    "quảng ninh": (21.006882, 107.292512),
    "hải dương": (20.937300, 106.314600),
    "hưng yên": (20.646082, 106.056312),
    "vĩnh phúc": (21.306082, 105.606312),
    "thừa thiên huế": (16.463712, 107.590862),
    "huế": (16.463712, 107.590862),
    "khánh hòa": (12.238772, 109.196749),
    "nha trang": (12.238772, 109.196749),
    "lâm đồng": (11.940412, 108.458312),
    "đà lạt": (11.940412, 108.458312),
    "bình định": (13.782000, 109.219400),
    "quy nhơn": (13.782000, 109.219400),
    "quảng nam": (15.568312, 108.480862),
    "quảng ngãi": (15.120862, 108.792312),
    "nghệ an": (18.673312, 105.681312),
    "thanh hóa": (19.806882, 105.785112),
    "tiền giang": (10.420412, 106.363712),
    "bến tre": (10.233312, 106.376862),
    "kiên giang": (10.012000, 105.080862),
    "an giang": (10.383312, 105.433312),
    "phú thọ": (21.320882, 105.228312),
    "thái nguyên": (21.592882, 105.844312),
    "tây ninh": (11.310412, 106.098712),
    "bình thuận": (10.933312, 108.100000),
    "gia lai": (13.983312, 108.000000),
    "đắk lắk": (12.666667, 108.050000),
}

# Tọa độ mặc định (Mốc Trung tâm Việt Nam)
DEFAULT_CENTER = (16.0, 106.0)

def get_coords_from_address(address, add_jitter=False):
    """
    Trích xuất Vĩ độ (lat), Kinh độ (lon), Mức Zoom và Tên vùng từ chuỗi địa chỉ text.
    
    Args:
        address: Chuỗi địa chỉ của doanh nghiệp
        add_jitter: Thêm độ lệch ngẫu nhiên nhỏ để các biểu tượng không đè lên nhau khi vẽ bản đồ
        
    Returns:
        tuple: (lat, lon, zoom_level, region_name)
    """
    if not address or pd.isna(address) or str(address).strip().lower() in ["n/a", "none", "", "nan"]:
        return DEFAULT_CENTER[0], DEFAULT_CENTER[1], 5, "Khác"
        
    addr_lower = str(address).lower()
    for key, (lat, lon) in VN_COORDS.items():
        if key in addr_lower:
            region_name = key.title()
            if add_jitter:
                jitter_lat = lat + np.random.uniform(-0.03, 0.03)
                jitter_lon = lon + np.random.uniform(-0.03, 0.03)
                return jitter_lat, jitter_lon, 14, region_name
            return lat, lon, 14, region_name
            
    return DEFAULT_CENTER[0], DEFAULT_CENTER[1], 5, "Khác"