import os
from pathlib import Path
import yaml
import pickle

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class RawDataset:

    def __init__(self, path: str):
        assert os.path.exists(path), "Đường dẫn không tồn tại"
        self.df = pd.read_csv(path)
        self.df["Ngày"] = pd.to_datetime(self.df["Ngày"], format="%m/%d/%Y %I:%M:%S %p")
        self._update_items()

        del self.df["Loại_tiền"]
        del self.df["Đơn_vị_tính"]
        del self.df["Ngành_hàng"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def _update_items(self):
        self.items = self.df["Tên_mặt_hàng"].unique()

    def _remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def _remove_null(self):
        self.df.dropna(inplace=True)

    def _remove_outlier(self):
        outlier_mask = pd.Series(False, index=self.df.index)

        for item in self.items:
            item_df = self.df[self.df["Tên_mặt_hàng"] == item]
            prices = item_df["Giá"]
            q1 = prices.quantile(0.25)
            q3 = prices.quantile(0.75)
            iqr = q3 - q1
            min_thresh = max(1000, q1 - 1.5 * iqr)
            max_thresh = q3 + 1.5 * iqr
            outlier = (self.df["Tên_mặt_hàng"] == item) & (
                (self.df["Giá"] < min_thresh) | (self.df["Giá"] > max_thresh)
            )
            outlier_mask |= outlier

        self.df = self.df[~outlier_mask]
        self._update_items()

    def _adf_filter(self):
        result_rows = []

        for item in self.items:
            prices = self.df.loc[self.df["Tên_mặt_hàng"] == item, "Giá"].values

            if len(prices) < 3:
                continue
            if np.all(prices == prices[0]):
                continue

            try:
                adf_result = adfuller(prices)
                result_rows.append({
                    "Tên_mặt_hàng": item,
                    "adf": adf_result[0],
                    "p-value": adf_result[1],
                    "n_lags": adf_result[2],
                    "n_obs": adf_result[3],
                    "1%": adf_result[4]["1%"],
                    "5%": adf_result[4]["5%"],
                    "10%": adf_result[4]["10%"]
                })
            except Exception:
                continue

        results_df = pd.DataFrame(result_rows)

        # Lọc ra những mặt hàng không ổn định
        filtered_items = results_df.loc[
            (results_df["p-value"] >= 0.05) &
            (results_df["adf"] >= results_df["1%"])
        ]["Tên_mặt_hàng"]

        self.df = self.df[self.df["Tên_mặt_hàng"].isin(filtered_items)]
        self._update_items()


    def _one_hot_encoding(self):
        cat_cols = self.df.drop(columns=["Ngày", "Giá", "Tên_mặt_hàng"], errors='ignore').columns

        for col in cat_cols:
            one_hot = pd.get_dummies(self.df[col], prefix=col)
            self.df = pd.concat([self.df.drop(columns=[col]), one_hot], axis=1)


    def _label_encoding(self):
        lbl_encoder = LabelEncoder()

        cat_cols = self.df.drop(columns=["Ngày", "Giá", "Tên_mặt_hàng"], errors='ignore').columns
        for col in cat_cols:
            self.df[col] = lbl_encoder.fit_transform(self.df[col])
                

    def _mm_normalizing(self):
        mm_normalizer = MinMaxScaler()

        cat_cols = self.df.drop(columns=["Ngày", "Giá", "Tên_mặt_hàng"]).columns
        for col in cat_cols:
            self.df[col] = mm_normalizer.fit_transform(self.df[[col]])


    def preprocess_all(self, oh_encoding=False, b4_oh_scaler=None, after_oh_scaler=None):
        self._remove_null()
        self._remove_duplicates()
        self._remove_outlier()
        self._adf_filter()

        # Lưu item và scaler
        self.get_item_metadata("data/metadata/item.yaml")
        self.get_scaler_metadata("data/metadata/scaler.yaml")

        if b4_oh_scaler:
            print("B4 OH scaler working")
            self.df.to_csv(b4_oh_scaler, index=False)

        if oh_encoding:
            self._one_hot_encoding()
        else:
            self._label_encoding()
            self._mm_normalizing()

        if after_oh_scaler:
            print("after oh scaler working")
            self.df.to_csv(after_oh_scaler, index=False)


    def get_item_metadata(self, to_yaml: str = None) -> dict:
        metadata = {
            item: {
                "n_rows": len(item_df),
                "id": idx,
                "first_update": item_df["Ngày"].min().strftime("%d/%m/%Y"),
                "last_update": item_df["Ngày"].max().strftime("%d/%m/%Y")
            }
            for idx, item in enumerate(self.items)
            if not (item_df := self.df[self.df["Tên_mặt_hàng"] == item]).empty
        }

        if to_yaml:
            yaml_path = Path(to_yaml)
            assert yaml_path.parent.exists(), "Đường dẫn gốc không tồn tại"
            with open(to_yaml, "w", encoding="utf-8") as file:
                yaml.dump(metadata, file, allow_unicode=True, sort_keys=False)

        return metadata
    

    def get_scaler_metadata(self, to_yaml: str=None):
        metadata = dict()

        for col in ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]:
            metadata.update({
                col: {
                    value: idx for idx, value in enumerate(self.df[col].unique())
                }
            })

            metadata[col].update({
                "max": len(self.df[col].unique()) -1
            })

        if to_yaml:
            with open(to_yaml, "w", encoding="utf-8") as file:
                yaml.safe_dump(metadata, stream=file, allow_unicode=True, sort_keys=False)

        return metadata


if __name__ == "__main__":
    dataset = RawDataset("data/data.csv")
    print(f"Before preprocessing: {len(dataset)} rows")

    metadata = dataset.get_item_metadata("data/metadata/raw.yaml")

    # LBL + MM
    dataset.preprocess_all(b4_oh_scaler="data/pre_data.csv", after_oh_scaler="data/scaler.csv")
    print(f"After preprocessing: {len(dataset)} rows")

    # OH encoding
    new_dataset = RawDataset("data/data.csv")
    new_dataset.preprocess_all(True, after_oh_scaler="data/oh.csv")