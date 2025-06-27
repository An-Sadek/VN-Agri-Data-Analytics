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


class RawDataset:

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

        # Tìm q1, q3, iqr, min_val, max_val của toàn bộ giá của dataset
        try:
            gia_toanbo = self.data["Giá"]

            self.q1 = np.quantile(gia_toanbo, 0.25)
            self.q3 = np.quantile(gia_toanbo, 0.75)
            self.iqr = self.q3 - self.q1
            self.min_val = self.q1 - 1.5 * self.iqr
            self.max_val = self.q3 + 1.5 * self.iqr

        except:
            print("Dataset không thể đọc được cột \"Giá\"")

        # Chuyển về dạng thời gian
        str2date_fn = lambda x: dt.strptime(x, "%m/%d/%Y %I:%M:%S %p")
        self.data["Ngày"] = self.data["Ngày"].apply(str2date_fn)

        # Lưu thuộc tính
        self.outlier_infos, self.outlier_df = self.get_outlier_infos()
        

    def __len__(self):
        return len(self.data)
    

    def get_stats(self, array: ARRAY) -> dict:
        n = len(array)
        min_val = np.min(array).item()
        max_val = np.max(array).item()
        mean = np.mean(array).item()
        median = np.median(array).item()
        mode = stats.multimode(array)
        std = np.std(array).item()
        var = np.var(array).item()

        try:
            mode = [x.item() for x in mode]
        except:
            pass

        return {
            "n": n,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "median": median,
            "mode": mode,
            "std": std,
            "var": var
        }
    

    def get_items_stats(self) -> dict:
        items = self.data["Tên_mặt_hàng"].unique()
        result = dict()

        for idx, item in enumerate(items):
            item_price = self.data[self.data["Tên_mặt_hàng"] == item]["Giá"]
            item_stats = self.get_stats(item_price)
            result.update({
                idx: item_stats
            })

        return result
    

    def remove_outlier(self)->None:
        """
        Chỉnh các giá trị dưới ngưỡng về max(1000, alpha)
        và các giá trị trên ngưỡng xuống beta
        """
        for idx in range(len(self.items)):
            if self.outlier_infos[idx]["have_outliers"]:
                alpha = self.outlier_infos[idx]["min_threshold"]
                beta = self.outlier_infos[idx]["max_threshold"]
                item = self.items[idx]

                self.data = self.data[~(
                    (self.data["Tên_mặt_hàng"] == item) & (
                        (self.data["Giá"] < max(1000, alpha)) |
                        (self.data["Giá"] > beta)
                    )
                )]

        self.outlier_infos = self.outlier_infos

    
    def update_all(self, fn: Callable[[Any], Any]) -> None:
        self.data["Giá"] = self.data["Giá"].apply(lambda x: fn(x))


    def get_outlier_infos(self) -> Union[dict, Tuple[dict, pd.DataFrame]]:
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
        

    def get_outlier_n_perc(self) -> Tuple[float, float]:
        """
        Hàm trả về số lượng ngoại lai và phần trăm ngoại lai.
        Tính theo giá từng sản phẩm
        """
        n_outlier_row = len(self.outlier_df)
        n_outlier_perc = n_outlier_row/len(self.data)

        return (n_outlier_row, n_outlier_perc)


    def get_item_metadata(self) -> dict:
        """
        Trả về các metadata của từng sản phẩm
        """
        item_metadata = dict()

        items_stats = self.get_items_stats()
        items_outliers = self.outlier_infos
        
        for idx, _ in enumerate(self.items):
            name = self.items[idx]
            item_metadata.update({
                idx: {
                    "name": name,
                    "first_update": np.min(self.data[self.data["Tên_mặt_hàng"] == name]["Ngày"])
                        .strftime("%Y/%m/%d"),
                    "last_update": np.max(self.data[self.data["Tên_mặt_hàng"] == name]["Ngày"])
                        .strftime("%Y/%m/%d"),
                    "stats": items_stats[idx],
                    "outliers": items_outliers[idx]
                }
            })

        return item_metadata


    def get_colmetadata(self) -> dict:
        # Thêm thông tin của toàn thể dataset
        ## Thêm thông tin cơ bản: id, name, type
        metadata = dict()
        colnames = self.data.columns

        for idx, colname in enumerate(colnames):
            metadata.update({colname: {
                "id": idx,
                "name": colname,
                "desc": None,
                "type": self.data[colname].dtype,
                "range/n_values": None,
                "data": None
            }})

        ## Đổi lại type cho dễ đọc
        for colname in colnames:
            col_dtype = metadata[colname]["type"]

            if col_dtype == np.dtype('O'):
                metadata[colname]["type"] = "str"

            elif col_dtype == np.dtype('float64'):
                metadata[colname]["type"] = "float"

            elif metadata[colname]["type"] == np.dtype('<M8[ns]'):
                metadata[colname]["type"] = "datetime"

            else:
                metadata[colname]["type"] = None

        ## Thêm miền giá trị/số lượng, và thông tin về cột
        for colname in colnames:

            ### Metadata của chuỗi
            if metadata[colname]["type"] == "str":
                """
                Đối với cột là str dữ liệu sẽ là từ điển của từ điển (dic[dict[int]]). 
                Với mỗi phần tử là thông tin của giá trị bao gồm:
                    name: là tên của giá trị độc nhất
                    n: là số lượng của giá trị độc nhất đó.
                {
                    0: {
                        "name": name0,
                        "n": n0
                    },
                    1: {
                        "name": name1,
                        "n": n1
                    },
                    ...
                }

                """
                str_data = dict()
                str_values = self.data[colname].unique()
                metadata[colname]["range/n_values"] = len(str_values)

                for idx, name in enumerate(str_values):
                    str_data.update({idx:{
                        "name": name,
                        "n": len(self.data[self.data[colname] == name])
                    }})

                metadata[colname]["data"] = str_data


            ### Metadata của dữ liệu liên tục
            if metadata[colname]["type"] == "float":
                """
                Đối với cột là dữ liệu số sẽ là 1 từ điển gồm các thông tin bao gồm: min, max, mean, median, mode, std, var, q1, q3, iqr, alpha, beta, số lượng ngoại lai, phần trăm ngoại lai
                """
                # Thêm thuộc tính 'stats' thống kê
                metadata[colname].update({"data":None})
                
                min_val = self.data[colname].min().item()
                max_val = self.data[colname].max().item()
                mean_val = np.mean(self.data[colname]).item()
                median_val = np.median(self.data[colname]).item()
                mode_val = self.data[colname].mode().to_list()
                std = np.std(self.data[colname]).item()
                var = np.var(self.data[colname]).item()

                #### Miền giá trị
                metadata[colname]["range/n_values"] = [min_val, max_val]

                #### Các thống kê
                metadata[colname]["data"] = {
                    "n": len(self.data),
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "median": median_val,
                    "mode": mode_val,
                    "std": std,
                    "var": var,
                }

            ### Metadata của dữ liệu thời gian
            if metadata[colname]["type"] == "datetime":
                f_update = np.min(self.data[colname]).strftime('%Y/%m/%d')
                l_update = np.max(self.data[colname]).strftime('%Y/%m/%d')
                metadata[colname]["range/values"] = [f_update, l_update]
                del metadata[colname]["data"]

        return metadata


    def get_item_df(self, 
        idx: int=None,  # type: ignore
        name: str =None # type: ignore
    ) -> pd.DataFrame:
        """
        Trả về các df chứa toàn bộ df của tên/idx cho trước
        """
        assert (idx is None) ^ (name is None), "Phải có idx hoặc name"

        if name is None:
            name = self.items[idx]

        return self.data[self.data["Tên_mặt_hàng"] == name]
    


if __name__ == "__main__":
    dataset = RawDataset("../../data/Rau, qua")
    print(dataset.data.shape)
    print(dataset.q1)
    print(dataset.q3)
    print(dataset.iqr)
    print(dataset.min_val)
    print(dataset.max_val)

    # Kiểm tra metadata tổng thể
    metadata = dataset.get_colmetadata()
    print(metadata)

    with open('file.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)

    # Thống kê từng mặt hàng
    items_stats = dataset.get_items_stats()

    with open('items_stats.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(items_stats, f, allow_unicode=True, sort_keys=False)

    # Lấy dữ liệu ngoại lai
    outlier_infos, outlier_df = dataset.get_outlier_infos()
    with open('outlier_infos.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(outlier_infos, f, allow_unicode=True, sort_keys=False)
    outlier_df.to_csv('./outlier_info.csv')

    items = dataset.data["Tên_mặt_hàng"].unique()
    outlier_items = []
    for idx in outlier_infos.keys():
        if outlier_infos[idx]["have_outliers"]:
            outlier_items.append(items[idx])
    print(len(outlier_items))
    print(outlier_items)

    # Lấy metadata item
    item_metadata = dataset.get_item_metadata()
    with open('item_metadata.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(item_metadata, f, allow_unicode=True, sort_keys=False)

    # Xoá giá trị ngoại lai
    dataset.remove_outlier()
    new_stats = dataset.get_items_stats()
    print(len(dataset))
    with open('new_stats.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(new_stats, f, allow_unicode=True, sort_keys=False)