import quandl

quandl_api_key = 'DxswD4b7N-71v3B7DaXk'
quandl.ApiConfig.api_key = quandl_api_key

symbols = [
    'AAPL','MMM', 'AXP', 'BA', 'CAT',
    'CVX', 'CSCO', 'KO', 'DD', 'XOM',
    'GS', 'HD', 'IBM', 'INTC', 'JNJ',
    'JPM', 'MCD', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'UNH', 'UTX', 'TRV',
    'VZ', 'V', 'WMT', 'WBA', 'DIS',
]

df_components = quandl.get((wiki_symbols := [f'WIKI/{s}' for s in symbols]),
                          start_date='2017-01-01',
                          end_date='2017-12-31',
                          column_index = 11)
df_components.columns = symbols

#%% Normalize the dataset
filled_df_components = df_components.fillna(method='ffill')
daily_df_components = filled_df_components.resample('24h').ffill()
daily_df_components = daily_df_components.fillna(method='bfill')

#%% Get DJIA dataset (Set up with YAHOO!)
import yfinance as yf

df = yf.download('^DJI',actions = 'inline' ,progress=False)
#%%
print(df.head())

#%% Convert to analysis material
import pandas as pd

# prepare the DataFrame
df_dji = pd.DataFrame(df['Adj Close'])
df_dji.columns = ['DJIA']
df_dji.index = pd.to_datetime(df_dji.index)

#trim the new dataframe and resample
djia_2017 = pd.DataFrame(df_dji.loc['2017-01-01':'2017-12-31'])
djia_2017 = djia_2017.resample('24h').ffill()

#%% Normalize dataset
from sklearn.decomposition import KernelPCA

fn_z_score = lambda x: (x - x.mean()) / x.std()

df_z_components = daily_df_components.apply(fn_z_score)
fitted_pca = KernelPCA().fit(df_z_components)

#%% Plot eigenvalues
%matplotlib inline
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)
plt.plot(fitted_pca.lambdas_)
plt.ylabel('Eigenvalues')
plt.show();

# See the first eigenvalues explain much of the variance of the data
#%% Obtaining weighted average values of eigenvalues

fn_weighted_avg = lambda x: x / x.sum()
weighted_values = fn_weighted_avg(fitted_pca.lambdas_)[:5]
print(weighted_values)
print(weighted_values.sum())

# Prints: [0.64863002 0.13966718 0.05558246 0.05461861 0.02313883]
        # 0.9216371041932275

#%% Reconstructing the Dow index with PCA
import numpy as np

kernel_pca = KernelPCA().fit(df_z_components)
pca_5 = kernel_pca.transform(daily_df_components)

weights = fn_weighted_avg(kernel_pca.lambdas_)
reconstructed_values = np.dot(pca_5, weights)

# Combine DJIA and PCA index for comparison
df_combined = djia_2017.copy()
df_combined['pca_5'] = reconstructed_values
df_combined = df_combined.apply(fn_z_score)
df_combined.plot(figsize=(12, 8));

#%% Analyzing a time series with trends

df = quandl.get('CHRIS/CME_GC1',
                column_index=6,
                collapse = 'monthly',
                start_date='2000-01-01')

#%% Compute the rolling mean and standard deviation

df_settle = df['Settle'].resample('MS').ffill().dropna()

df_rolling = df_settle.rolling(12)
df_mean = df_rolling.mean()
df_std = df_rolling.std()

#%% Visualize the plot of the rolling mean
plt.figure(figsize=(12, 8))
plt.plot(df_settle, label='Orginal')
plt.plot(df_mean, label='mean')
plt.legend();

#%% Visualize std
df_std.plot(figsize=(12, 8));

#%% Perform an ADF unit root test
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_settle)
print('ADF statistic: ', result[0])
print('p-value: ', result[1])

critical_values = result[4]
for key, value in critical_values.items():
    print(f'Critical value {key}: {value:.2f}')
# PRINTS: ADF statistic:  -0.36422183600515456
        # p-value:  0.9159032507747725
        # Critical value 1%: -3.46
        # Critical value 5%: -2.87
        # Critical value 10%: -2.57
#%% Detrending
df_log = np.log(df_settle)

df_log_ma = df_log.rolling(2).mean()
df_detrend = df_log - df_log_ma
df_detrend.dropna(inplace=True)

df_detrend_rolling = df_detrend.rolling(12)
df_detrend_ma = df_detrend_rolling.mean()
df_detrend_std = df_detrend_rolling.std()

# Plot
plt.figure(figsize=(12, 8))
plt.plot(df_detrend, label='Detrended')
plt.plot(df_detrend_ma, label='Mean')
plt.plot(df_detrend_std, label='std')
plt.legend(loc='upper right');

#%% ADF test:
#%% Perform an ADF unit root test
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_detrend)
print('ADF statistic: ', result[0])
print('p-value: ', result[1])

critical_values = result[4]
for key, value in critical_values.items():
    print(f'Critical value {key}: {value:.2f}')
# PRINTS: ADF statistic:  -17.68818391773999
        # p-value:  3.5823189763193525e-30
        # Critical value 1%: -3.46
        # Critical value 5%: -2.87
        # Critical value 10%: -2.57

#%% Removing trend by differencing
df_log_diff = df_log.diff(periods=3).dropna()

# Mean and standard deviation of differenced data
df_diff_rolling = df_log_diff.rolling(12)
df_diff_ma = df_diff_rolling.mean()
df_diff_std = df_diff_rolling.std()

# Plot the stationary data
plt.figure(figsize=(12, 8))
plt.plot(df_log_diff, label='Differenced')
plt.plot(df_diff_ma, label='mean')
plt.plot(df_diff_std, label='std')
plt.legend(loc='upper right');


#%% Perform an ADF
from statsmodels.tsa.stattools import adfuller

result = adfuller(df_log_diff)

print('ADF statistic:', result[0])
print('p-value: %.5f' % result[1])

critical_values = result[4]
for key, value in critical_values.items():
    print('Critical value (%s): %.3f' % (key, value))
# PRINTS: ADF statistic: -2.9870722580313194
        # p-value: 0.03612
        # Critical value (1%): -3.459
        # Critical value (5%): -2.874
        # Critical value (10%): -2.573

#%% Seasonal decomposing
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_result = seasonal_decompose(df_log.dropna(), period = 12)
df_trend = decompose_result.trend
df_season = decompose_result.seasonal
df_residual = decompose_result.resid

plt.rcParams["figure.figsize"] = (12, 8)
fig = decompose_result.plot();

#%% Perform an ADF
result = adfuller(df_residual.dropna())

print('ADF statistic:',  result[0])
print('p-value: %.5f' % result[1])

critical_values = result[4]
for key, value in critical_values.items():
    print('Critical value (%s): %.3f' % (key, value))
# PRINTS: ADF statistic: -6.872153208131203
        # p-value: 0.00000
        # Critical value (1%): -3.460
        # Critical value (5%): -2.875
        # Critical value (10%): -2.574

#%% Finding SARIMA model paramets by grid search
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

def arima_grid_search(df, s):
    p = d = q = range(2)
    param_combinations = list(itertools.product(p, d, q))
    lowest_aic, pdq, pdqs = None, None, None
    total_iterations = 0
    for order in param_combinations:
        for (p, q, d) in param_combinations:
            seasonal_order = (p, q, d, s)
            total_iterations += 1
            try:

                model = SARIMAX(df, order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    disp=False)
                model_result = model.fit(maxiter=200, disp=False)

                if not lowest_aic or model_result.aic < lowest_aic:

                    lowest_aic = model_result.aic
                    pdq, pdqs = order, seasonal_order
            except Exception as ex:
                continue
    return lowest_aic, pdq, pdqs

# get best SARIMA model
lowest_aic, order, seasonal_order = arima_grid_search(df_settle, 12)
print(f'ARIMA{order}x{seasonal_order}')
print(f'Lowest AIC: {lowest_aic:.3f}')
# PRINTS: ARIMA(0, 1, 1)x(0, 1, 1, 12)
        # Lowest AIC: 2426.192

#%% Fitting the model
model = SARIMAX(df_settle,order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
            disp=False)

model_results = model.fit(maxiter=200, disp=False)
# print(model_results.summary())
model_results.plot_diagnostics(figsize=(12, 8));
model_results.resid.describe()

#%% Predicting and forecasting the SARIMAX model
n = len(df_settle.index)
prediction = model_results.get_prediction(start=n-12*5,end=n+5)
prediction_ci = prediction.conf_int()

# Plot the predicted and forecasted prices
plt.figure(figsize=(12, 8))

ax = df_settle['2008':].plot(label='actual')
prediction_ci.plot(
    ax=ax, style=['--', '--'],
    label = 'predicted/forecasted'
)

ci_index = prediction_ci.index
lower_ci = prediction_ci.iloc[:, 0]
upper_ci = prediction_ci.iloc[:, 1]

ax.fill_between(ci_index, lower_ci, upper_ci,
    color='r', alpha=.1)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Prices')

plt.legend()
plt.show();
