from stocker import Stocker
import quandl
import matplotlib.pyplot as plt

# Quandl API key
quandl.ApiConfig.api_key = "ykjGjY8xpsA1_sM1Y84q"


# MSFT is in the WIKI database, which is default
microsoft = Stocker(ticker='MSFT')

# TECHM is in the NSE database
techm = Stocker(ticker='TECHM', exchange='NSE')

microsoft.plot_stock(start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic')

microsoft.buy_and_hold(start_date=None, end_date=None, nshares=1)

model, future = microsoft.create_prophet_model(days=200, resample=False)

model.plot_components(future) 
