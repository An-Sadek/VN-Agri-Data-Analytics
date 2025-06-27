import os
from typing import Callable, Any, Union, Tuple

from datetime import datetime as dt

import yaml
import numpy as np
import pandas as pd
import statistics as stats


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


    def get_col_metadata(self) -> dict:
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
    

    def get_metadata(self) -> dict:
        """
        Trả về metadata của tập dữ liệu
        """
        metadata = dict()

        metadata.update({
            "src": "https://thitruongnongsan.gov.vn/vn/nguonwmy.aspx",
            "len": len(self),
            "number_of_columns": self.data.shape[1],
            "stats": self.get_stats(self.data["Giá"]),
            "columns": self.get_col_metadata(),
            "items": self.get_item_metadata()
        })
        
        return metadata


if __name__ == "__main__":
    dataset = RawDataset("data/Rau, qua")
    print(dataset.data.shape)

    # Kiểm tra thống kế của từng sản phẩm
    item_metadata = dataset.get_item_metadata()
    with open("test/item_metadata.yaml", "w", encoding='utf-8') as file:
        yaml.dump(item_metadata, file, encoding="utf8-", sort_keys=False, allow_unicode=True)

    # Kiểm tra metadata của cột
    col_metadata = dataset.get_col_metadata()
    with open("test/col_metadata.yaml", "w", encoding='utf-8') as file:
        yaml.dump(col_metadata, file, encoding="utf8-", sort_keys=False, allow_unicode=True)

    # Kiểm tra metadata của tập dữ liệu
    metadata = dataset.get_metadata()
    with open("test/metadata.yaml", "w", encoding='utf-8') as file:
        yaml.dump(metadata, file, encoding="utf8-", sort_keys=False, allow_unicode=True)