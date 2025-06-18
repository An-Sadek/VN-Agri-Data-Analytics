import os
from datetime import datetime as dt
from typing import Callable, Optional, Any

import yaml
import json
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


    def __len__(self):
        return len(self.data)


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

        print(metadata)

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
                metadata[colname].update({"stats":None})
                
                min_val = self.data[colname].min().item()
                max_val = self.data[colname].max().item()
                mean_val = np.mean(self.data[colname]).item()
                median_val = np.median(self.data[colname]).item()
                mode_val = self.data[colname].mode().to_list()
                std = np.std(self.data[colname]).item()
                var = np.var(self.data[colname]).item()

                q1 = np.quantile(self.data[colname], 0.25).item()
                q3 = np.quantile(self.data[colname], 0.75).item()
                iqr = q3 - q1
                alpha = q1 - 1.5 * iqr
                beta = q3 + 1.5 * iqr
                n_outlier = len(
                    self.data[
                        (self.data[colname] < 1000) |
                        (self.data[colname] < alpha) |
                        (self.data[colname] > beta)
                    ]
                )
                outlier_perc = n_outlier/len(self.data)

                #### Miền giá trị
                metadata[colname]["range/n_values"] = [min_val, max_val]

                #### Các thống kê
                metadata[colname]["stats"] = {
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "median": median_val,
                    "mode": mode_val,
                    "std": std,
                    "var": var,
                    
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "min_threshold": alpha,
                    "max_threshold": beta,
                    "n_outlier": n_outlier,
                    "outlier_perc": outlier_perc
                }

            ### Metadata của dữ liệu thời gian
            if metadata[colname]["type"] == "datetime":
                datetime_data = {
                    "first_update": np.min(self.data[colname]).strftime('%Y/%m/%d'),
                    "last_update": np.max(self.data[colname]).strftime('%Y/%m/%d')
                }

                metadata[colname]["data"] = datetime_data

        return metadata

  


    def get_outlier_infos(self):
        pass
    
    
    def get_outlier_mathang(self):
        """
        Chỉ trả về tên của các mặt hàng sau khi được xác định có giá trị ngoại lai 
        dựa trên giá của sản phẩm cụ thể.
        """
        pass
    
    
    def get_outlier_mathang_df(self, 
            name: str, 
            min_val: int, 
            max_val: int
    ) -> pd.DataFrame:
        """
        Lấy những dòng mà chứa giá trị ngoại lai của mặt hàng.

        Đầu vào:
            name: Tên mặt hàng đang xét
            min_val: Cận dưới hợp lệ đã được tính từ tứ phân vị
            max_val: Cấn trên hợp lệ đã được tính từ tứ phân vị

        Trả về: DataFrame chứa các dòng bị ngoại lai
        """
        outlier_df = self.data[
                (self.data["Tên_mặt_hàng"] == name) & (
                (self.data["Giá"] < 1000) |
                (self.data["Giá"] < min_val) |
                (self.data["Giá"] > max_val)
            )
        ]

        return outlier_df
    
    
    def change_outlier_values_df(self, 
            name: str, 
            min_val: int, 
            max_val: int, 
            fn: Callable[[Any], Any] = lambda x: None,
            value = None,
            inplace = False
    ) -> pd.DataFrame:
        """

        """
        assert (fn(0) is not None) ^ (value is not None), "Phải có hàm hoặc là giá trị thế vào. Và chỉ có 1 trong 2"

        outlier_df = self.get_outlier_mathang_df(name, min_val, max_val)
        new_df = outlier_df.copy()
        
        if fn(0) is not None:
            new_df["Giá"] = outlier_df["Giá"].apply(fn)

        if value is not None:
            new_df["Giá"] = value

        updated_df = self.data.copy()
        updated_df.loc[new_df.index, "Giá"] = new_df["Giá"]

        if inplace:
            self.data = updated_df

        return updated_df
    
    def remove_outlier(self, name: str, min_val: int, max_val: int)->None:
        self.data = self.data[~(
                (self.data["Tên_mặt_hàng"] == name) & (
                (self.data["Giá"] < min_val) |
                (self.data["Giá"] > max_val)
            )
        )]

    def calc_outlier_perc(self, name: str, min_val, max_val: int) -> float:
        outlier_df = self.get_outlier_mathang_df(name, min_val, max_val)
        n_outlier = len(outlier_df)
        n_mathang = len(self.data[self.data["Tên_mặt_hàng"] == name])
        return n_outlier / n_mathang


    def plot(self, names: tuple[str]|str, row: int, col: int, figsize=(20, 20)) -> None:
        """
        Hàm vẽ các plot theo tên được chỉ định. 
        Dùng để đánh giá sơ bộ các giá trị bị ngoại lai.

        Đầu vào:
            names: Tuple gồm các phần tử là tên các mặt hàng được yêu cầu
            row: Số nguyên các hàng muốn hiển thị
            col: Số nguyên các cột muốn hiển thị
        """
        

        if isinstance(names, tuple):
            assert len(names) == (row * col), "Số hàng và cột không khớp với kích thước tên các mặt hàng"
            _, axes = plt.subplots(row, col, figsize=figsize)
            i = j = 0

            for name in names:
                array = self.data[self.data["Tên_mặt_hàng"] == name]["Giá"].values
                axes[i, j].plot(array)
                axes[i, j].set_title(name)

                j += 1
                if j % col == 0:
                    j = 0
                    i += 1

        if isinstance(names, str):
            array = self.data[self.data["Tên_mặt_hàng"] == names]["Giá"]
            plt.plot(array)

        plt.show()
        plt.close()


    def plot_one(self, name: str, figsize=(10, 5)) -> None:
        price = self.data[self.data["Tên_mặt_hàng"] == name]["Giá"]
        date = self.data[self.data["Tên_mặt_hàng"] == name]["Ngày"]

        date = pd.to_datetime(date)
        unit = self.data[self.data["Tên_mặt_hàng"] == name]["Đơn_vị_tính"].unique()[0]

        # Plot
        plt.figure(figsize=figsize)
        plt.plot(date, price)

        # Chỉnh lại chỉ có ngày
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to every year
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Show only the year

        # Giới hạn 2020 -> hiện tại
        plt.xlim(pd.to_datetime("2020-01-01"), pd.to_datetime("2025-06-15"))

        # Improve layout
        plt.xlabel("Năm")
        plt.ylabel(f"Giá ({unit})")
        plt.title(f"{name}")
        plt.grid(True)
        plt.tight_layout()

        plt.show()


    def show_boxplot(self, name: str) -> None:
        """
        Hiển thị boxplot giá của mặt hàng

        Đầu vào:
            name: Tên mặt hàng đang xét
        """
        try:
            sns.boxplot(self.data[self.data["Tên_mặt_hàng"] == name])
            plt.title("{0} ({1})".format(
                    name, 
                    self.data[self.data["Tên_mặt_hàng"] == name]["Đơn_vị_tính"].unique()[0]
                )
            )
            plt.show()
            plt.close()
        except:
            print("Sai tên mặt hàng?")


if __name__ == "__main__":
    dataset = VNAgriDataset("../../data/Rau, qua")
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

    # Kiểm tra metadata từng mặt hàng
