# us-electricity-prod-forecast
# Forecasting US Electric Production: Statistical vs. Deep Learning Models

> A comparative time-series study evaluating classical statistical models (**SARIMA**, **Holt-Winters**) against deep learning architectures (**LSTM**, **1D CNN**) on 30+ years of monthly US electric production data.

---

### 1. Objective
To develop, tune, and benchmark statistical and neural forecasting models on a non-stationary, strongly seasonal energy dataset—identifying whether algorithmic complexity outperforms structural model-data alignment on moderately sized monthly time series.

### 2. Data Source
* **File:** `Electric_Production.csv` (397 monthly observations, 0 missing values)
* **Time Span:** January 1985 – January 2018
* **Schema:**
  * `DATE` — Monthly timestamp (`MM/DD/YYYY`)
  * `EP` — US Electric Production (IP) Index
* **Summary Statistics (`EP`):**
  * **Mean / Median:** `88.85` / `89.78` (Std Dev: `15.39`)
  * **Range:** `55.32` (May 1985) to `129.40` (January 2018)
* **Characteristics:** Persistent upward secular trend with a rigid 12-month seasonal cycle whose amplitude scales with the trend.

### 3. Tools & Libraries
* **Language:** `Python` (Jupyter Notebook)
* **Data & Visualization:** `pandas`, `NumPy`, `matplotlib`
* **Statistical Modelling:** `statsmodels` (Seasonal Decomposition, ADF Test, SARIMAX, Holt-Winters)
* **Deep Learning & Evaluation:** `TensorFlow / Keras` (LSTM, 1D CNN), `scikit-learn` (MinMaxScaler, RMSE, MAE)

### 4. Methodology
* **Exploratory Analysis & Stationarity:**
  * Applied *multiplicative seasonal decomposition* to isolate trend, seasonality, and residuals.
  * Confirmed non-stationarity via the *Augmented Dickey-Fuller (ADF) test* (ADF = `-2.2570`, *p* = `0.1862`) and achieved stationarity through first-order differencing (ADF = `-7.1049`, *p* = `0.0000`).
* **Statistical Models:**
  * **SARIMA:** Conducted an AIC-based grid search to select the optimal specification `(1, 1, 2) x (0, 1, 1, 12)` and verified white-noise residuals via ACF/PACF, Q-Q plots, and Ljung-Box diagnostics.
  * **Holt-Winters:** Implemented multiplicative exponential smoothing to capture level, trend, and proportional seasonal variance.
* **Deep Learning Models:**
  * Scaled the series to `[0, 1]` using `MinMaxScaler` and framed supervised learning with a 12-month lookback window (`L = 12`).
  * Trained a single-layer **LSTM** (50 units, ReLU) and a **1D CNN** (64 filters, kernel size 2, max-pooling, 50-unit dense layer) using Adam and MSE loss.

### 5. Model Performance (Unseen Test Set)

| Model | Category | RMSE | MAE |
| :--- | :--- | :---: | :---: |
| **SARIMA** | Statistical | **4.1092** | **3.2004** |
| **Holt-Winters** | Statistical | 5.2530 | 4.2264 |
| **1D CNN** | Deep Learning | 5.2732 | 3.7420 |
| **LSTM** | Deep Learning | 5.3565 | 3.8986 |

*Note: Statistical models were evaluated on the final 36 months; deep learning models were evaluated on the aligned 24-month subset due to the 12-month lookback window. SARIMA also outperformed a Naïve Baseline MAE of `15.2425`.*

### 6. Key Insights
* **Model-Data Alignment Wins:** **SARIMA** achieved the lowest RMSE (`4.1092`) and MAE (`3.2004`), accurately tracking both seasonal troughs and extreme production peaks.
* **Holt-Winters Sensitivity:** Captured the annual rhythm well but tended to overshoot later peaks near the upper end of the historical range.
* **Deep Learning Constraints:** Both **LSTM** and **1D CNN** learned the general cyclic trajectory but produced flatter forecasts that under-predicted extreme seasonal peaks—highlighting that parameter-heavy neural networks are less data-efficient than seasonal statistical models on ~400 monthly observations.
