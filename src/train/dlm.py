import numpy as np
import pandas as pd


class DLM:
    def __init__(self):
        """
        Ý nghĩa từng ma trận
            A:  Ma trận chuyển trạng thái xác định hệ thống tuyến tính giữa 2 trạng thái. 
                Tùy thuộc vào bài toán cụ thể thì ma trận chuyển trạng thái sẽ khác nhau.
            H:  Ma trận H liên kết trạng thái ẩn (state) với quan sát thực tế (measurement). 
                Nó mô tả cách trạng thái x_t  được chuyển đổi thành giá trị quan sát z_t 
            Q:  Ma trận Q biểu thị mức độ không chắc chắn (uncertainty) trong quá trình tiến hóa của trạng thái,
                do nhiễu quá trình (process noise). Trong mô hình Kalman Filter, 
                nhiễu quá trình được giả định là nhiễu trắng Gauss với hiệp phương sai Q w_t ~ N(0, Q)
            R:  Ma trận R biểu thị mức độ không chắc chắn trong các quan sát thực tế, do nhiễu đo lường (measurement noise). 
                Trong Kalman Filter, nhiễu đo lường được giả định là nhiễu trắng Gauss với hiệp phương sai R
            H:  Ma trận P biểu thị mức độ không chắc chắn (uncertainty) trong ước lượng trạng thái x_t . 
                Nó là hiệp phương sai của trạng thái ước lượng tại thời điểm hiện tại.
            B: Ma trận B chứa các hệ số liên kết các biến ngoại sinh (exogenous variables) với quan sát
        """
        self.A = np.array([[1.0]])     # Ma trận chuyển trạng thái (State transition matrix)
        self.H = np.array([[1.0]])     # Ma trận quan sát (Observation matrix)
        self.Q = np.array([[0.001]])   # Ma trận hiệp phương sai nhiễu quá trình (Process noise covariance)
        self.R = np.array([[0.01]])    # Ma trận hiệp phương sai nhiễu đo lường (Measurement noise covariance)
        self.P = np.zeros((1, 1))      # Ma trận hiệp phương sai ước lượng trạng thái
        self.x = None                  # Trạng thái ước lượng hiện tại (Current state estimate)
        self.B = None                  # Ma trận hệ số cho biến ngoại sinh (Exogenous coefficients)

    def fit(self, y, exog):
        """
        Phương thức dùng để fit vào các tham số
            y: Giá trị muốn dự đoán
            exog: (Các) biến ngoại sinh
        """
        self.x = np.array([[y[0]]]) # Giá trị khởi tạo trạng thái ban đầu bằng quan sát đầu tiên
        y = np.asarray(y).reshape(-1, 1)  # Đưa dữ liệu thành vector cột
        T = len(y)  # Số bước thời gian

        # Nếu không có biến ngoại sinh → tạo ma trận rỗng (T × 0)
        if exog is None:
            exog = np.zeros((T, 0))
        else:
            exog = np.asarray(exog)

        k = exog.shape[1]  # Số biến ngoại sinh
        self.B = np.zeros((k, 1))  # Khởi tạo hệ số ngoại sinh

        # Nếu có biến ngoại sinh → ước lượng hệ số B bằng pseudo-inverse
        if k > 0:
            self.B = np.linalg.pinv(exog) @ y  # (k × 1)
            y_adj = y - exog @ self.B  # Loại bỏ ảnh hưởng của biến ngoại sinh khỏi quan sát
        else:
            y_adj = y  # Không có biến ngoại sinh thì giữ nguyên

        # Vòng lặp cập nhật Kalman Filter
        for t in range(T):
            # Dự báo (Prediction step)
            x_pred = self.A @ self.x  # Dự báo trạng thái tiếp theo
            P_pred = self.A @ self.P @ self.A.T + self.Q  # Dự báo hiệp phương sai

            # Cập nhật (Update step)
            z_t = y_adj[t]  # Quan sát đã điều chỉnh
            S = self.H @ P_pred @ self.H.T + self.R  # Hiệp phương sai của sai số dự đoán
            K = P_pred @ self.H.T @ np.linalg.inv(S)  # Hệ số Kalman (Kalman gain)
            self.x = x_pred + K @ (z_t - self.H @ x_pred)  # Cập nhật trạng thái
            self.P = (np.eye(1) - K @ self.H) @ P_pred     # Cập nhật hiệp phương sai

        return self

    def forecast(self, exog_future: np.ndarray, steps=1):
        """
        Dự báo 'steps' bước tiếp theo dựa trên mô hình đã fit.
            exog_future: ma trận (steps x k) chứa giá trị biến ngoại sinh trong tương lai
            steps: Số bước, tương ứng với số dòng của exog_future
        """
        forecasts = []
        x_fore = self.x.copy()  # Trạng thái ban đầu để dự báo

        for t in range(steps):
            # Trạng thái tiến hóa (không thêm nhiễu khi dự báo)
            x_fore = self.A @ x_fore
            # Thêm ảnh hưởng của biến ngoại sinh
            exog_term = 0
            if self.B is not None and exog_future.size > 0:
                exog_term = exog_future[t] @ self.B
            forecasts.append(float(self.H @ x_fore + exog_term))

        return np.array(forecasts)
    
    
if __name__ == "__main__":
    item_df = pd.read_csv("data/pre_data.csv")
    item_df = item_df[item_df["Tên_mặt_hàng"] == 23]
    X = item_df[["Thị_trường", "Loại_giá"]]
    y = item_df["Giá"]

    model = DLM()
    model.fit(y, X)


