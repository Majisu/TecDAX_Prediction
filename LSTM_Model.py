# importing libraries and moduls

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.losses import mean_squared_error
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

TecDAX_data = pd.read_csv("TECDAX.csv")                        # load TecDAX from 2009-05-01 till 2019-05-01
TecDAX_data["Date"] = pd.to_datetime(TecDAX_data["Date"])      # convert 'Date' from object to datetime

TecDAX_data.set_index("Date", inplace=True)                  # set index       
TecDAX_data.info()
TecDAX_data.sort_index(inplace=True)                         # sort index
