import os
import yaml
from datetime import datetime as dt

import pandas as pd
import numpy as np
import pickle

import warnings

warnings.filterwarnings("ignore")


class ForcastModel:

    def __init__(self):
        pass


    def forcast_by_date(
            ngay,
            ten_mat_hang,
            thi_truong,
            loai_gia,
            nguon,
            model_type: str,
            encoding_type: str
    ):
        assert model_type in ["dlm", "sarimax"]
        assert encoding_type in ["oh", "lbl"]


if __name__ == "__main__":
    pass