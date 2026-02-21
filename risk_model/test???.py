import pandas as pd
import matplotlib.pyplot as plt

vols = pd.read_csv("garch_vols.csv", index_col=0, parse_dates=True)

print(vols.shape)     # e.g. (2500, 5) -> 2500 days × 5 assets
print(vols.head())    # see the first few days

vols["SPY"].plot(figsize=(10, 4), title="SPY GARCH(1,1) Conditional Volatility")
plt.show()