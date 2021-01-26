# tafel_fitter


Example:

```python
import numpy as np
import pandas as pd
from tafel_fitter import tafel

df = pd.read_csv("Input_files/Pt-HER_100mM_perchloric_acid_2500rpm.csv")
x = df["Overpotential / V"].values
y = df["<I>/A"].values

i = -3
x = x[:i]
y = y[:i]

results = tafel.fit_all(x, y)
results.head()
d = tafel.filter_r2(results)
best_fit, subset = tafel.find_best_fit(d)

plt.plot(x, np.log10(np.abs(y)))
plt.plot(x, np.log10(best_fit["j0"]) + best_fit["dlog(j)/dV"]*x, color="k")
plt.axvline(best_fit["window_min"], color='k', alpha=0.5)
plt.axvline(best_fit["window_max"], color='k', alpha=0.5)
plt.xlabel("overpotential (V)")
plt.ylabel("log current")
plt.show()
```