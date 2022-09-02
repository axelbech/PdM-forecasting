#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ThymeBoost import ThymeBoost as tb

#%%
np.random.seed(100)
trend = np.linspace(1, 50, 100) + 50
seasonality = ((np.cos(np.arange(1, 101))*10))
exogenous = np.random.randint(low=0, high=2, size=len(trend))
y = trend + seasonality + exogenous * 20
#reshape for 2d to pass to ThymeBoost
exogenous = exogenous.reshape(-1, 1)

#%%
trend = np.linspace(1, 100, 100)
def sigmoid(index):
    return (1 / (1 + np.exp(-0.05*(index-50)))) + (1 / (1 + np.exp(-0.5*(index-100)))) + np.sin(0.1*index)
trend = sigmoid(trend)
seasonality = ((np.cos(np.arange(1, 101))*0))
y = trend

#%%
boosted_model = tb.ThymeBoost(verbose=1)

output = boosted_model.fit(y,
                           trend_estimator='fast_ets',
                           seasonal_estimator='classic',
                           exogenous_estimator='ols',
                           global_cost='maicc',
                           fit_type='global')


# %%
#create a future exogenous input
forecast_horizon = 50
# np.random.seed(100)
# future_exogenous = np.random.randint(low=0, high=2, size=forecast_horizon)
#use predict method and pass fitted output, forecast horizon, and future exogenous
predicted_output = boosted_model.predict(output,
                                         forecast_horizon=forecast_horizon)
boosted_model.plot_results(output, predicted_output)

# %%
