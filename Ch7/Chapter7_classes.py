import yfinance as yf

df_vix = yf.download('^VIX',actions = 'inline', start='2000-01-03',
            end='2018-12-22', progress=False)
df_spx = yf.download('^GSPC', actions = 'inline', start='2000-01-03',
            end='2018-12-22', progress=False)

#%%
df_spx.info()

#%% Create Dataframe
import pandas as pd

df = pd.DataFrame({
    'SPX': df_spx['Adj Close'],
    'VIX': df_vix['Adj Close']
})
df.index = pd.to_datetime(df.index)
#%% Describe
%matplotlib inline
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

ax_spx = df['SPX'].plot()
ax_vix = df['VIX'].plot(secondary_y=True)

ax_spx.legend(loc=1)
ax_vix.legend(loc=2)

plt.show();

#%% The set of differences between the prior period values
df.diff().hist(
    figsize=(10, 5),
    color='blue',
    bins=100
);
# OR:
df.pct_change().hist(
    figsize=(10, 5),
    color='blue',
    bins=100
);
#%% Plot logarithm values
import numpy as np

log_returns = np.log(df / df.shift(1)).dropna()
log_returns.plot(
    subplots=True,
    figsize=(10, 8),
    color='blue',
    grid=True
);
for ax in plt.gcf().axes:
    ax.legend(loc='upper left')

#%% Correlations:
log_returns.corr()

plt.ylabel('Rolling Annual Correlation')

df_corr = df['SPX'].rolling(252).corr(other=df['VIX'])
df_corr.plot(figsize=(12, 8));
