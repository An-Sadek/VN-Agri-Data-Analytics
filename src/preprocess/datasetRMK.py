import os
from typing import Callable, Optional, Any, Union, Tuple

import matplotlib.dates as mdates
from datetime import datetime as dt

import yaml
import numpy as np
import pandas as pd
import statistics as stats

import seaborn as sns
import matplotlib.pyplot as plt


ARRAY = np.ndarray|list|tuple|pd.Series


class VNAgriDataset:

    def __init__(self, path: str):
        """
        path: Đường dẫn dẫn đến file html được lấy từ csdl nông sản
        """
        
        # Đọc đường dẫn
        assert os.path.exists(path), "Đường dẫn đến file không tồn tại"
        self.path = path

        # Đọc file html và chuyển về dạng bảng
        with open(path, "r", encoding="utf-16") as file:
            html = file.read()
        self.data = pd.read_html(html)[0]
        self.items = self.data["Tên_mặt_hàng"].unique().tolist()

        # Chuyển về dạng thời gian
        str2date_fn = lambda x: dt.strptime(x, "%m/%d/%Y %I:%M:%S %p")
        self.data["Ngày"] = self.data["Ngày"].apply(str2date_fn)

        # Luư thông tin về outlier và thống kê
        self.item_metadata = self.get_item_metadata()


    def __len__(self):
        return len(self.data)
    

    def get_item_stats(self):
        """
        """
        items_stats = dict()

        for idx, item in enumerate(self.items):
            price = self.data[self.data["Tên_mặt_hàng"] == item]["Giá"]
            items_stats.update({
                idx: {
                    "n": len(price),
                    "min": np.min(price).item(),
                    "max": np.max(price).item(),
                    "mean": np.mean(price).item(),
                    "median": np.median(price).item(),
                    "mode": stats.multimode(price),
                    "std": np.std(price).item(),
                    "var": np.var(price).item()
                }
            })

        return items_stats
    
    
    def get_outlier_infos(self):
        """
        Hàm kiểm tra các số dòng ngoại lai
        """
        items = self.data["Tên_mặt_hàng"].unique()
        item_outlier_info = dict()
        
        outlier_dfs = pd.DataFrame()

        for idx, item in enumerate(items):
            item_price = self.data[self.data["Tên_mặt_hàng"] == item]["Giá"]
            q1 = np.quantile(item_price, 0.25).item()
            q3 = np.quantile(item_price, 0.75).item()
            iqr = q3 - q1
            alpha = q1 - 1.5 * iqr
            beta = q3 + 1.5 * iqr

            outlier_df = self.data[
                (self.data["Tên_mặt_hàng"] == item) & (
                    (self.data["Giá"] < 1000) |
                    (self.data["Giá"] < alpha) |
                    (self.data["Giá"] > beta)
                )
            ]
            n_outlier = len(outlier_df)
            outlier_perc = n_outlier/len(self.data)

            item_outlier_info.update({
                idx: {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "min_threshold": alpha,
                    "max_threshold": beta,
                    "have_outliers": n_outlier > 0,
                    "n_outlier": n_outlier,
                    "outlier_perc": outlier_perc
                }
            })

            outlier_dfs = pd.concat([outlier_dfs, outlier_df], axis=0)

        return (item_outlier_info, outlier_dfs)
    

    def get_item_metadata(self):
        """
        """
        outlier_infos, _ = self.get_outlier_infos()
        items_stats = self.get_item_stats()
        item_metadata = dict()

        for idx, item in enumerate(self.items):
            item_metadata.update({
                idx: {
                    "name": item,
                    "stats": items_stats[idx],
                    "outlier_infos": outlier_infos[idx]
                }
            })

        return item_metadata


    def update_outlier(self):
        """
        """
        for idx, item in enumerate(self.items):
            pass


if __name__ == "__main__":
    dataset = VNAgriDataset("../../data/Rau, qua")

    # Lấy ngoại lai
    outlier_infos, _ = dataset.get_outlier_infos()
    with open('../../test/outlier_infos.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(outlier_infos, f, allow_unicode=True, sort_keys=False)
    print(outlier_infos[0]["have_outliers"])

    # Lấy thống kê
    item_stats = dataset.get_item_stats()
    with open('../../test/item_stats.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(item_stats, f, allow_unicode=True, sort_keys=False)
    print(item_stats[0])

    # Lấy metadata của vật sản phẩm
    item_metadata = dataset.item_metadata
    with open('../../test/item_metadata.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(item_metadata, f, allow_unicode=True, sort_keys=False)
    print(item_metadata[0])