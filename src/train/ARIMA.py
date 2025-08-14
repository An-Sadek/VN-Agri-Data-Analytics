import numpy as np
import pandas as pd


class ARIMA:

    def __init__(self, data, d, p, q):
        assert p < len(data), "p must be smaller than data length"

        self.data = np.array(data, dtype=float)
        self.d = d
        self.p = p
        self.q = q


    def AR(self):
        # Build DataFrame with target and lags
        df = pd.DataFrame({'y': self.data})
        for i in range(1, self.p + 1):
            df[f'y_lag{i}'] = df['y'].shift(i)

        df = df.dropna()

        # X = lag values, y = target
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        # Add constant term (intercept)
        X = np.column_stack((np.ones(len(X)), X))

        # OLS estimation of coefficients
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

        #
        y_hat = X @ beta
        residuals = y - y_hat

        return (beta, residuals)
    
    def difference(self, d):
        """Difference the series d times."""
        diffed = np.array(self.data, dtype=float)
        for _ in range(d):
            diffed = diffed[1:] - diffed[:-1]
        return diffed
    

    def MA(self):
        # Step 1: Get residuals from AR part if p>0, else just use mean
        if self.p > 0:
            _, residuals = self.AR()
        else:
            mean_y = np.mean(self.data)
            residuals = self.data - mean_y
        
        # Step 2: Build DataFrame with target and lagged residuals
        df = pd.DataFrame({'y': self.data})
        for i in range(1, self.q + 1):
            df[f'e_lag{i}'] = pd.Series(residuals).shift(i)
        
        df = df.dropna()
        
        # X = lagged errors, y = target
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        # Add constant (mu)
        X = np.column_stack((np.ones(len(X)), X))

        # OLS estimate of theta's
        theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        
        y_hat = X @ theta
        new_residuals = y - y_hat
        
        return theta, new_residuals
    
    def forecast(self, steps=1):
        """Forecast the next `steps` values."""
        
        # Step 1: Work with differenced data if needed
        if self.d > 0:
            data = self.difference(self.d)
        else:
            data = self.data.copy()
        
        # Step 2: Fit AR and MA parts
        beta, ar_res = self.AR() if self.p > 0 else (np.array([np.mean(data)]), np.zeros(len(data)))
        theta, ma_res = self.MA() if self.q > 0 else (np.array([0]), np.zeros(len(data)))

        # Step 3: Prepare initial values for simulation
        y_hist = list(data)
        e_hist = list(ma_res)

        forecasts = []

        # Step 4: Generate forecasts step-by-step
        for _ in range(steps):
            # AR term
            ar_part = beta[0]  # intercept
            for i in range(1, self.p + 1):
                ar_part += beta[i] * y_hist[-i]

            # MA term
            ma_part = 0
            if self.q > 0:
                for j in range(1, self.q + 1):
                    ma_part += theta[j] * e_hist[-j]

            # Predicted next value
            y_next = ar_part + ma_part
            forecasts.append(y_next)

            # Append predicted value and a zero residual (future residuals are unknown)
            y_hist.append(y_next)
            e_hist.append(0)

        # Step 5: Reverse differencing if needed
        if self.d > 0:
            # Last original value(s) for integration
            orig_values = self.data[-self.d:]
            for i in range(len(forecasts)):
                forecasts[i] = forecasts[i] + orig_values[-1]
                orig_values.append(forecasts[i])

        return forecasts
    

if __name__ == "__main__":
    y = np.array([4.0, 4.5, 5.0, 5.3, 5.7, 6.1, 6.4, 6.8])
    arima_model = ARIMA(y, d=0, p=2, q=0)
    ar_coefs, residuals = arima_model.AR()
    print("Intercept:", ar_coefs[0])
    print("AR coefficients:", ar_coefs[1:])
    print("AR res: ", residuals)

    # MA model
    ma_coefs, ma_res = arima_model.MA()
    print("MA res: ", ma_res)

    # Forecast
    results = arima_model.forecast(5)
    print(results)
