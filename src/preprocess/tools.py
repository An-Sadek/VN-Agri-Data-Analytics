import os

import numpy as np
import pandas as pd


class OutlierTool:

    def __init__(self, path: str):
        """
        path: Đường dẫn dẫn đến file html được lấy từ csdl nông sản
        """
        
        # Đọc đường dẫn
        assert os.path.exists(path), "Đường dẫn đến file không tồn tại"
        self.path = path

        # Đọc file html và chuyển về dạng bảng
        with open("../../data/Ca phe", "r", encoding="utf-16") as file:
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
    

    def get_outlier_mathang(self) -> tuple:
        """
        Những mặt hàng có thể có giá riêng và cần được xem xét riêng. Lọc ra các mặt hàng vẫn có giá trị ngoại lai dựa trên giá mặt hàng đó.
        """

