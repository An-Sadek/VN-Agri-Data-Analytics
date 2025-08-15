import numpy as np
import pandas as pd

class ARIMAX:
    def __init__(self, data, p=1, d=0, q=0, exog=None, smoothing=True, alpha=0.3):
        assert p < len(data), "p must be smaller than data length"
        self.data = np.array(data, dtype=float)
        self.p = p
        self.d = d
        self.q = q

        if exog is not None:
            exog = np.array(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            assert exog.shape[0] == len(data), "exog must have same length as data"
            self.exog = exog
        else:
            self.exog = None

        if smoothing:
            self.data = self.exponential_smoothing(alpha)

    def get_data(self):
        return self.data
    
    def get_full(self, steps=1, exog_future=None):
        return np.concatenate([self.data, self.forecast(steps, exog_future)]).tolist()
    
    # ------------------------
    # I part - differencing
    # ------------------------
    def difference(self, data, d=1):
        diffed = np.array(data, dtype=float)
        for _ in range(d):
            diffed = diffed[1:] - diffed[:-1]
        return diffed

    def inverse_difference(self, last_values, forecasts):
        result = []
        prev = list(last_values)
        for f in forecasts:
            value = f + prev[-1]
            result.append(value)
            prev.append(value)
        return result
    
    def exponential_smoothing(self, alpha):
        smoothed = [self.data[0]]
        for t in range(1, len(self.data)):
            smoothed.append(alpha * self.data[t] + (1 - alpha) * smoothed[t-1])
        return np.array(smoothed)

    # ------------------------
    # AR part
    # ------------------------
    def AR(self, data, exog=None):
        df = pd.DataFrame({'y': data})
        for i in range(1, self.p + 1):
            df[f'y_lag{i}'] = df['y'].shift(i)

        # Add exog if given
        if exog is not None:
            for k in range(exog.shape[1]):
                df[f'exog{k}'] = exog[:, k]

        df = df.dropna()
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        X = np.column_stack((np.ones(len(X)), X))  # intercept
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        y_hat = X @ beta
        residuals = y - y_hat
        return beta, residuals

    # ------------------------
    # MA part
    # ------------------------
    def MA(self, data, residuals):
        if self.q == 0:
            return np.array([0]), np.zeros(len(data))

        df = pd.DataFrame({'y': data})
        for i in range(1, self.q + 1):
            df[f'e_lag{i}'] = pd.Series(residuals).shift(i)

        df = df.dropna()
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        X = np.column_stack((np.ones(len(X)), X))
        theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        y_hat = X @ theta
        new_residuals = y - y_hat
        return theta, new_residuals

    # ------------------------
    # Forecast method
    # ------------------------
    def forecast(self, steps=1, exog_future=None):
        # 1. Difference data if needed
        if self.d > 0:
            y_train = self.difference(self.data, self.d)
        else:
            y_train = self.data

        # Prepare exog for training
        if self.exog is not None:
            exog_train = self.exog
            if self.d > 0:
                exog_train = exog_train[self.d:]
        else:
            exog_train = None

        # 2. Fit AR
        if self.p > 0:
            beta, ar_res = self.AR(y_train, exog_train)
        else:
            beta = np.array([np.mean(y_train)])
            ar_res = np.zeros(len(y_train))

        # 3. Fit MA
        theta, ma_res = self.MA(y_train, ar_res)

        # 4. Prepare history
        y_hist = list(y_train)
        e_hist = list(ma_res)

        # Prepare exog future
        if self.exog is not None:
            assert exog_future is not None, "exog_future must be provided when using exogenous variables"
            exog_future = np.array(exog_future, dtype=float)
            if exog_future.ndim == 1:
                exog_future = exog_future.reshape(-1, 1)
            assert exog_future.shape[0] == steps, "exog_future must have shape (steps, n_features)"
        else:
            exog_future = None

        forecasts_diff = []

        # 5. Step-by-step forecast
        for step in range(steps):
            ar_part = beta[0]  # intercept

            # AR lags
            for i in range(1, self.p + 1):
                ar_part += beta[i] * y_hist[-i]

            # exog part
            if self.exog is not None:
                for k in range(exog_future.shape[1]):
                    ar_part += beta[self.p + 1 + k] * exog_future[step, k]

            # MA part
            ma_part = 0
            if self.q > 0:
                for j in range(1, self.q + 1):
                    ma_part += theta[j] * e_hist[-j]

            y_next = ar_part + ma_part
            forecasts_diff.append(y_next)

            y_hist.append(y_next)
            e_hist.append(0)  # future residuals assumed zero

        # 6. Convert back to original scale
        if self.d > 0:
            last_vals = self.data[-self.d:]
            forecasts = self.inverse_difference(last_vals, forecasts_diff)
        else:
            forecasts = forecasts_diff

        return forecasts


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    y = np.array([4.0, 4.5, 5.0, 5.3, 5.7, 6.1, 6.4, 6.8])
    exog = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # one exogenous variable
    model = ARIMAX(y, p=1, d=1, q=1, exog=exog)

    exog_future = np.array([9, 10, 11, 12, 13])
    preds = model.forecast(5, exog_future=exog_future)
    print("Forecasts:", preds)

    full_hist = model.get_full(5, exog_future=exog_future)
    print(full_hist)
