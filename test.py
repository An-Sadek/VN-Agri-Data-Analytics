import pandas as pd
import numpy as np

# Create example data
np.random.seed(42)

dates = pd.to_datetime([
    "2023-01-01", "2023-01-05", "2023-01-10", "2023-01-15", "2023-01-20",
    "2023-02-01", "2023-02-03", "2023-02-15", "2023-03-01", "2023-03-10"
])
n = len(dates)

df = pd.DataFrame({
    "date": dates,
    "y": np.random.randn(n).cumsum() + 10,
    "x1": np.random.randn(n),
    "x2": np.random.rand(n) * 5
})

# Simple fix - just fit the model directly
from orbit.models.dlt import DLT

model = DLT(
    response_col="y",
    date_col="date",
    regressor_col=["x1", "x2"],
    seed=42
)

# Don't use Forecaster - just fit directly
model.fit(df)

# Make predictions
predictions = model.predict(df)
print(predictions)