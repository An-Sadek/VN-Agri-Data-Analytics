import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

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

    def exponential_smoothing(self, alpha):
        smoothed = [self.data[0]]
        for t in range(1, len(self.data)):
            smoothed.append(alpha * self.data[t] + (1 - alpha) * smoothed[t-1])
        return np.array(smoothed)

    # ------------------------
    # AR + exog
    # ------------------------
    def AR(self, data, exog=None):
        y_series = pd.Series(data)
        columns = {'y': y_series}
        for i in range(1, self.p + 1):
            columns[f'y_lag{i}'] = y_series.shift(i)
        df = pd.DataFrame(columns)

        if exog is not None:
            exog_df = pd.DataFrame(exog, columns=[f'exog{i}' for i in range(exog.shape[1])])
            df = pd.concat([df, exog_df], axis=1)

        df = df.dropna()
        y = df['y'].values
        X = df.drop(columns=['y']).values
        X = np.column_stack((np.ones(len(X)), X))

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        residuals = y - y_hat
        return beta, residuals

    # ------------------------
    # MA part
    # ------------------------
    def MA(self, data):
        if self.q == 0:
            return np.array([]), data

        e_series = pd.Series(data)
        columns = {'e': e_series}
        for i in range(1, self.q + 1):
            columns[f'e_lag{i}'] = e_series.shift(i)
        df = pd.DataFrame(columns)

        df = df.dropna()
        y = df['e'].values
        X = df.iloc[:, 1:].values  # no intercept

        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ theta
        new_residuals = y - y_hat
        return theta, new_residuals

    # ------------------------
    # Forecast method
    # ------------------------
    def forecast(self, steps=1, exog_future=None):
        if exog_future is not None:
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
        theta, ma_res = self.MA(ar_res)

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
                    ma_part += theta[j-1] * e_hist[-j]

            y_next = ar_part + ma_part
            forecasts_diff.append(y_next)
            y_hist.append(y_next)
            e_hist.append(0)

        # 6. Inverse difference if d > 0
        if self.d == 0:
            return np.array(forecasts_diff)

        # Compute last differences at each level
        diff_histories = []
        temp = self.data
        for _ in range(self.d):
            diff = np.diff(temp)
            if len(diff) == 0:
                raise ValueError("Data too short for differencing")
            diff_histories.append(diff[-1])
            temp = diff

        # Prepare add_list
        add_list = list(reversed(diff_histories[:-1])) if self.d > 1 else []
        add_list += [self.data[-1]]

        # Cumulatively undifference
        forecasts = np.array(forecasts_diff)
        for i in range(self.d):
            forecasts = np.cumsum(forecasts) + add_list[i]

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


class Tools:
    def __init__(self):
        pass

    def find_d(self, data, max_d=5, significance=0.1):
        diffed = np.array(data, dtype=float)
        
        for d in range(max_d + 1):
            result = adfuller(diffed)
            p_value = result[1]
            
            if p_value < significance:
                return d
            # difference for next iteration
            diffed = diffed[1:] - diffed[:-1]
        
        return max_d
    
    def find_p(self, data, d=1, max_p=5, exog=None):
        best_p = 0
        min_aic = np.inf

        if d > 0:
            diffed_data = ARIMAX(data, d=d).difference(data, d=d)
            if exog is not None:
                exog = exog[d:]  
        else:
            diffed_data = np.array(data)

        n = len(diffed_data)
        max_p = min(n // 2, max_p)

        for p in range(max_p + 1):
            if p >= n:
                break
            model = ARIMAX(diffed_data, p=p, d=0, q=1, exog=exog) 
            _, ar_res = model.AR(diffed_data)
            rss = np.sum(ar_res ** 2)
            n_obs = len(ar_res)
            if rss < 1e-10:  # Perfect fit
                aic = 2 * (p + 1)
            else:
                aic = n_obs * np.log(rss / n_obs + 1e-10) + 2 * (p + 1)

            if aic < min_aic:
                min_aic = aic
                best_p = p

        return best_p
    

    def find_q(self, data, p=1, d=1, max_q=5, exog=None):
        best_q = 0
        min_aic = np.inf

        # 1. difference the data if d > 0
        if d > 0:
            diffed_data = ARIMAX(data, d=d, exog=exog).difference(data, d=d)
            if exog is not None:
                exog = exog[d:]  # Slice exog to match diffed_data length
        else:
            diffed_data = np.array(data)

        n = len(diffed_data)
        max_q = min((len(data) - d - p) // 2, max_q)

        # 2. loop over q values
        for q in range(max_q + 1):
            if p + q >= n:
                break
            model = ARIMAX(diffed_data, p=p, d=0, q=q, exog=exog)  # d=0 because data already differenced

            _, ar_res = model.AR(diffed_data)

            # Fit MA for current q
            _, ma_res = model.MA(ar_res)
            rss = np.sum(ma_res ** 2)
            n_obs = len(ma_res)
            if rss < 1e-10:
                aic = 2 * (p + q + 1)  # Adjust params: AR (p+1), MA (q)
            else:
                aic = n_obs * np.log(rss / n_obs + 1e-10) + 2 * (p + q + 1)

            if aic < min_aic:
                min_aic = aic
                best_q = q

        return best_q


    def find_best(self, data, max_order=(5, 5, 5), exog=None) -> dict:
        best_d = self.find_d(data, max_order[0], significance=0.1)
        best_p = self.find_p(data, best_d, max_order[1], exog=exog)
        best_q = self.find_q(data, best_p, best_d, max_order[2], exog=exog)

        return {
            "p": best_p,
            "d": best_d,
            "q": best_q
        }
    
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

    tools = Tools()
    best_hyperpara = tools.find_best(y)
    print(best_hyperpara)
    
    # Real data
    df = pd.read_csv("data/pre_data.csv")
    item_df = df[df["Tên_mặt_hàng"] == 23]

    exog = item_df[["Thị_trường", "Nguồn"]].values
    y = item_df["Giá"].values

    best_hyperpara = tools.find_best(y, exog=exog)
    new_model = ARIMAX(
        data = y,
        exog = exog,
        p = best_hyperpara["p"],
        d = best_hyperpara["d"],
        q = best_hyperpara["q"]
    )

    exog_future = [[19, 7]]
    exog_future = np.tile(exog_future, (20, 1))
    print(exog_future.shape)
    y_forecast = new_model.forecast(20, exog_future)
    print("Real data", y_forecast)