import numpy as np
from scipy.optimize import minimize

class ARIMA:
    def __init__(self, data, p=1, d=0, q=0):
        self.data = np.array(data, dtype=float)  # original time series
        self.p = p  # AR order
        self.d = d  # differencing order
        self.q = q  # MA order

    def _difference(self, series, d):
        """Subtract previous value d times (to make series stationary)."""
        for _ in range(d):
            series = series[1:] - series[:-1]
        return series

    def _log_likelihood(self, params, series):
        """Compute the negative log-likelihood for given AR/MA params."""
        p, q = self.p, self.q
        ar = params[:p]           # AR coefficients
        ma = params[p:p+q]        # MA coefficients
        mu = params[p+q]          # constant term
        sigma = params[-1]        # std deviation of residuals

        n = len(series)
        e = np.zeros(n)           # store errors
        y_hat = np.zeros(n)       # store predictions

        for t in range(n):
            # AR term: sum of AR coeffs * past y's
            ar_part = sum(ar[i] * series[t-i-1] for i in range(p) if t-i-1 >= 0)
            # MA term: sum of MA coeffs * past errors
            ma_part = sum(ma[j] * e[t-j-1] for j in range(q) if t-j-1 >= 0)
            # Predicted value
            y_hat[t] = mu + ar_part + ma_part
            # Error = actual - predicted
            e[t] = series[t] - y_hat[t]

        # Gaussian log-likelihood
        ll = -0.5 * n * np.log(2*np.pi*sigma**2) - (e @ e) / (2*sigma**2)
        return -ll  # We minimize, so return negative

    def fit(self):
        """Fit ARIMA model to data."""
        # Step 1: difference the series if needed
        series = self._difference(self.data, self.d) if self.d > 0 else self.data.copy()

        # Step 2: initial guess for parameters (all zeros, mean, std)
        init_params = np.r_[np.zeros(self.p), np.zeros(self.q), np.mean(series), np.std(series)]

        # Step 3: optimize parameters to maximize likelihood
        result = minimize(self._log_likelihood, init_params, args=(series,), method="BFGS")
        self.params_ = result.x
        return self

    def forecast(self, steps=1):
        """Forecast future values."""
        # Work on differenced series
        series = self.difference(self.data, self.d) if self.d > 0 else self.data.copy()
        p, q = self.p, self.q
        ar = self.params_[:p]
        ma = self.params_[p:p+q]
        mu = self.params_[p+q]

        # Store past values and residuals
        y_hist = list(series)
        e_hist = [0] * len(series)

        forecasts_diff = []

        for _ in range(steps):
            ar_part = sum(ar[i] * y_hist[-i-1] for i in range(p) if len(y_hist)-i-1 >= 0)
            ma_part = sum(ma[j] * e_hist[-j-1] for j in range(q) if len(e_hist)-j-1 >= 0)
            y_next = mu + ar_part + ma_part
            forecasts_diff.append(y_next)

            # Update history
            y_hist.append(y_next)
            e_hist.append(0)  # assume no future errors

        # Convert differenced forecast back to original scale
        if self.d > 0:
            return self.inverse_difference(self.data[-self.d:], forecasts_diff)
        else:
            return forecasts_diff


# Example usage
if __name__ == "__main__":
    y = [4.0, 4.5, 5.0, 5.3, 5.7, 6.1, 6.4, 6.8]
    model = ARIMA(y, p=1, d=0, q=1)
    model.fit()
    print("Fitted parameters:", model.params_)
    print("Forecast next 3 steps:", model.forecast(3))
