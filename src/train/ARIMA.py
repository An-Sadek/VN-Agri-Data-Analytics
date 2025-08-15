import numpy as np
import pandas as pd

class ARIMA:
    def __init__(self, data, p=1, d=0, q=0):
        assert p < len(data), "p must be smaller than data length"
        self.data = np.array(data, dtype=float)
        self.p = p
        self.d = d
        self.q = q

    # ------------------------
    # I part - differencing
    # ------------------------
    def difference(self, data, d=1):
        """Apply differencing d times to given data array."""
        diffed = np.array(data, dtype=float)
        for _ in range(d):
            diffed = diffed[1:] - diffed[:-1]
        return diffed

    def inverse_difference(self, last_values, forecasts, d=1):
        """Reverse differencing to get values back to original scale."""
        result = []
        prev = list(last_values)
        for f in forecasts:
            value = f + prev[-1]
            result.append(value)
            prev.append(value)
        return result

    # ------------------------
    # AR part
    # ------------------------
    def AR(self, data):
        """Fit AR(p) model using OLS."""
        df = pd.DataFrame({'y': data})
        for i in range(1, self.p + 1):
            df[f'y_lag{i}'] = df['y'].shift(i)

        df = df.dropna()
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        # Add intercept
        X = np.column_stack((np.ones(len(X)), X))

        # OLS estimation
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        y_hat = X @ beta
        residuals = y - y_hat
        return beta, residuals

    # ------------------------
    # MA part
    # ------------------------
    def MA(self, data, residuals):
        """Fit MA(q) model using residuals from AR part."""
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
    def forecast(self, steps=1):
        """Forecast the next `steps` values in original scale."""
        # 1. Difference data if needed
        if self.d > 0:
            y_train = self.difference(self.data, self.d)
        else:
            y_train = self.data

        # 2. Fit AR
        if self.p > 0:
            beta, ar_res = self.AR(y_train)
        else:
            beta = np.array([np.mean(y_train)])
            ar_res = np.zeros(len(y_train))

        # 3. Fit MA
        theta, ma_res = self.MA(y_train, ar_res)

        # 4. Prepare history for simulation
        y_hist = list(y_train)
        e_hist = list(ma_res)

        forecasts_diff = []

        # 5. Step-by-step forecast
        for _ in range(steps):
            # AR contribution
            ar_part = beta[0]  # intercept
            for i in range(1, self.p + 1):
                ar_part += beta[i] * y_hist[-i]

            # MA contribution
            ma_part = 0
            if self.q > 0:
                for j in range(1, self.q + 1):
                    ma_part += theta[j] * e_hist[-j]

            # Prediction in differenced space
            y_next = ar_part + ma_part
            forecasts_diff.append(y_next)

            # Update histories
            y_hist.append(y_next)
            e_hist.append(0)  # future residuals unknown, assume 0

        # 6. Convert back to original scale if differenced
        if self.d > 0:
            last_vals = self.data[-self.d:]
            forecasts = self.inverse_difference(last_vals, forecasts_diff, self.d)
        else:
            forecasts = forecasts_diff

        return forecasts


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    y = np.array([4.0, 4.5, 5.0, 5.3, 5.7, 6.1, 6.4, 6.8])
    model = ARIMA(y, p=2, d=1, q=1)

    preds = model.forecast(5)
    print("Forecasts:", preds)
