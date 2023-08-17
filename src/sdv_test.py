import numpy as np
from sdv.demo import load_timeseries_demo

# entity_columns: 'Symbol'
# context_columns: 'MarketCap', 'Sector', 'Industry'
# sequence_index: 'Date'
# data_columns: values to learn the PAR model


data = load_timeseries_demo()
print(data)
print(type(data))
