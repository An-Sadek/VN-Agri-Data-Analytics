import os
from datetime import datetime as dt
from typing import Callable, Optional, Any

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
        with open("../../data/Rau, qua", "r", encoding="utf-16") as file:
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


    def get_outlier_chung(self) -> tuple:
        """
        Hàm lấy tên các mặt hàng dựa trên các min_val và max_val đã tính ở hàm khởi tạo
        """
        outlier_df = self.data[(self.data["Giá"] < self.min_val) | (self.data["Giá"] > self.max_val)]

        try:
            outlier_mathang = tuple(outlier_df["Tên_mặt_hàng"].unique())
        except:
            print("Không thể đọc được cột \"Tên_mặt_hàng\"")

        return outlier_mathang
    

    def get_outlier_infos_one(self, name) -> tuple:
        mathang_df = self.data[self.data["Tên_mặt_hàng"] == name]
        gia_mathang = mathang_df["Giá"].values
        
        q1 = np.quantile(gia_mathang, 0.25)
        q3 = np.quantile(gia_mathang, 0.75)
        iqr = q3 - q1
        min_mathang = q1 - 1.5 * iqr
        max_mathang = q3 + 1.5 * iqr

        soluong_ngoailai = np.sum((mathang_df["Giá"] < min_mathang) | (mathang_df["Giá"] > max_mathang))

        return (name, q1, q3, iqr, min_mathang, max_mathang, soluong_ngoailai)
    

    def get_outlier_infos(self) -> tuple:
        """
        Những mặt hàng có thể có giá riêng và cần được xem xét riêng. 
        Lọc ra các mặt hàng vẫn có giá trị ngoại lai dựa trên giá mặt hàng đó.

        Trả về: Tuple gồm 6 phần tử bao gồm tên mặt hàng và các giá trị q1, q3, iqr, min_val, max_val
        """
        outlier_filtered = []
        outlier_mathang = self.get_outlier_chung()

        for value in outlier_mathang:
            infos = self.get_outlier_infos_one(value)

            if infos[-1] > 0:
                outlier_filtered.append(infos)

        outlier_filtered = tuple(outlier_filtered)

        return outlier_filtered
    
    
    def get_outlier_mathang(self) -> tuple[str]:
        """
        Chỉ trả về tên của các mặt hàng sau khi được xác định có giá trị ngoại lai 
        dựa trên giá của sản phẩm cụ thể.
        """
        ten_mathang = [x[0] for x in self.get_outlier_infos()]
        return tuple(ten_mathang)
    
    
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


    def plot(self, names: tuple[str], row: int, col: int, figsize=(20, 20)) -> None:
        """
        Hàm vẽ các plot theo tên được chỉ định. 
        Dùng để đánh giá sơ bộ các giá trị bị ngoại lai.

        Đầu vào:
            names: Tuple gồm các phần tử là tên các mặt hàng được yêu cầu
            row: Số nguyên các hàng muốn hiển thị
            col: Số nguyên các cột muốn hiển thị
        """
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
    outlier = VNAgriDataset("../../data/Rau, qua")
    print(outlier.data.shape)
    print(outlier.q1)
    print(outlier.q3)
    print(outlier.iqr)
    print(outlier.min_val)
    print(outlier.max_val)

    # Kiểm tra outlier tổng thể
    outlier_mathang = outlier.get_outlier_chung()
    print(len(outlier_mathang))
    print(outlier_mathang)

    # Kiểm tra outlier cục bộ
    outlier_filtered = outlier.get_outlier_mathang()
    print(len(outlier_filtered))
    print(outlier_filtered)

    # Lấy thông tin outlier
    print("\nOutlier infos")
    outlier_infos = outlier.get_outlier_infos()
    print(len(outlier_infos))
    names = [x[0] for x in outlier_infos]
    min_vals = [x[4] for x in outlier_infos]
    max_vals = [x[5] for x in outlier_infos]
    print(names)
    print(min_vals)
    print(max_vals)

    # Lấy df của sản phẩm có giá trị ngoại lai
    outlier_df0 = outlier.get_outlier_mathang_df(names[0], min_vals[0], max_vals[0])
    print(outlier_df0)

    # Thế df đã được xử lý
    ## Bằng hàm
    print("\n\nThế giá trị ngoại lai bằng hàm")
    replaced_fn_df = outlier.change_outlier_values_df(names[0], min_vals[0], max_vals[0], fn=lambda x: x*1000)
    print(replaced_fn_df[replaced_fn_df["Tên_mặt_hàng"]==names[0]]["Giá"].mean())
    plt.plot(replaced_fn_df[replaced_fn_df["Tên_mặt_hàng"] == names[0]]["Giá"].values)
    plt.show()

    ## Bằng hằng
    print("\n\nThế giá trị ngoại lai bằng hằng")
    replaced_value_df = outlier.change_outlier_values_df(
        names[0], 
        min_vals[0], 
        max_vals[0], 
        value=80000
    )
    print(replaced_value_df[replaced_value_df["Tên_mặt_hàng"]==names[0]]["Giá"].mean())
    plt.plot(replaced_value_df[replaced_value_df["Tên_mặt_hàng"] == names[0]]["Giá"].values)
    plt.show()
    
