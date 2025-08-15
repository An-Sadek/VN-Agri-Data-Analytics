import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class ARIMAX:
    def __init__(self, data, exog=None, p=1, d=0, q=0, smoothing=True, alpha=0.3):
        assert p < len(data), "p must be smaller than data length"
        self.data = np.array(data, dtype=float)
        self.exog = np.array(exog, dtype=float) if exog is not None else None
        if self.exog is not None:
            assert len(self.exog) == len(self.data), "Exogenous variables must match data length"
        self.p = p
        self.d = d
        self.q = q

        if smoothing:
            self.data = self.exponential_smoothing(alpha)

    def get_data(self):
        return self.data
    
    def get_full(self, steps=1, exog_future=None):
        forecasted = self.forecast(steps, exog_future)
        return np.concatenate([self.data, forecasted]).tolist()
    
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
    # AR + exog
    # ------------------------
    def AR(self, data, exog=None):
        df = pd.DataFrame({'y': data})
        for i in range(1, self.p + 1):
            df[f'y_lag{i}'] = df['y'].shift(i)

        if exog is not None:
            exog_df = pd.DataFrame(exog, columns=[f'exog{i}' for i in range(exog.shape[1])])
            df = pd.concat([df, exog_df], axis=1)

        df = df.dropna()
        y = df['y'].values
        X = df.drop(columns=['y']).values
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
        assert steps == len(exog_future)

        # 1. Difference
        if self.d > 0:
            y_train = self.difference(self.data, self.d)
        else:
            y_train = self.data

        # 2. Fit AR + exog
        exog_train = None
        if self.exog is not None:
            exog_train = self.exog[self.d:] if self.d > 0 else self.exog
        beta, ar_res = self.AR(y_train, exog_train)

        # 3. Fit MA
        theta, ma_res = self.MA(y_train, ar_res)

        # 4. Prepare history
        y_hist = list(y_train)
        e_hist = list(ma_res)
        forecasts_diff = []

        # 5. Forecast step by step
        for t in range(steps):
            ar_part = beta[0]  # intercept
            for i in range(1, self.p + 1):
                ar_part += beta[i] * y_hist[-i]

            # Exogenous contribution
            if self.exog is not None:
                if exog_future is None:
                    raise ValueError("Future exogenous data required for forecasting")
                exog_t = exog_future[t]
                for j, val in enumerate(exog_t):
                    ar_part += beta[self.p + 1 + j] * val

            ma_part = 0
            if self.q > 0:
                for j in range(1, self.q + 1):
                    ma_part += theta[j] * e_hist[-j]

            y_next = ar_part + ma_part
            forecasts_diff.append(y_next)
            y_hist.append(y_next)
            e_hist.append(0)

        if self.d > 0:
            last_vals = self.data[-self.d:]
            forecasts = self.inverse_difference(last_vals, forecasts_diff)
        else:
            forecasts = forecasts_diff

        return forecasts
    
def acf(y, nlags=20):
    y = np.array(y)
    n = len(y)
    mean_y = np.mean(y)
    var_y = np.var(y)
    acf_vals = []

    for lag in range(nlags + 1):
        cov = np.sum((y[lag:] - mean_y) * (y[:n-lag] - mean_y)) / n
        acf_vals.append(cov / var_y)
    return np.array(acf_vals)

# ------------------------
# Partial Autocorrelation function
# ------------------------
def pacf(y, nlags=20):
    from numpy.linalg import lstsq
    pacf_vals = [1.0]  # lag 0 PACF is 1
    y = np.array(y)

    for k in range(1, nlags + 1):
        # Build the lagged matrix
        X = np.column_stack([y[i:-(k-i)] for i in range(k)]) if k > 1 else y[:-1].reshape(-1,1)
        y_k = y[k:]
        beta, _, _, _ = lstsq(X, y_k, rcond=None)
        pacf_vals.append(beta[-1])  # last coefficient = PACF at lag k

    return np.array(pacf_vals)

# ------------------------
# Plotting function
# ------------------------
def plot_acf_pacf(y, nlags=20):

    if nlags is None:
        nlags = len(y) // 2

    assert len(y) >= nlags

    acf_vals = acf(y, nlags)
    pacf_vals = pacf(y, nlags)
    
    lags = np.arange(nlags+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)
    plt.stem(lags, acf_vals, basefmt=" ")
    plt.title("ACF")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=1.96/np.sqrt(len(y)), color='red', linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(y)), color='red', linestyle='--')

    plt.subplot(1,2,2)
    plt.stem(lags, pacf_vals, basefmt=" ")
    plt.title("PACF")
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=1.96/np.sqrt(len(y)), color='red', linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(len(y)), color='red', linestyle='--')

    plt.tight_layout()
    plt.show()


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    y = np.array([4.0, 4.5, 5.0, 5.3, 5.7, 6.1, 6.4, 6.8])
    exog = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # simple exogenous variable

    model = ARIMAX(y, exog=exog, p=0, d=1, q=1)
    
    # Forecasting next 3 steps with exogenous values
    exog_future = np.array([[9], [10], [11]])
    preds = model.forecast(3, exog_future)
    print("Forecasts:", preds)

    full_series = model.get_full(steps=3, exog_future=exog_future)
    plot_acf_pacf(full_series, nlags=None)
