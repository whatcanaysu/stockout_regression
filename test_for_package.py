# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 00:03:58 2022

@author: FDN-Aysu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
import datetime
from datetime import datetime, timedelta, time
from tqdm import tqdm
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col


from stock_in_out_regression import stockout_functions

pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None 


sales = pd.read_pickle("preprocessed_sales_updated.pkl")
stock = pd.read_pickle("product_availability_updated.pkl")
pr_available = stock.groupby('date')['is_available'].mean() # some days have all products not available (no info days)

stock = stock[~stock.product_id.isin(['Delivery_007','xxxx'])]
stock['product_id'] = pd.to_numeric(stock["product_id"])
stock = stock.sort_values(['product_id','date'])
stock = stock.groupby(['product_id','date']).head(1) # some (product,date) have different availability
stock = stock.loc[~stock.date.isin(pr_available[pr_available == 0].index),] # drop days without information
print('stock:', stock.date.min(), stock.date.max())

sales['date'] = pd.to_datetime(sales.order_datetime).dt.normalize()
sales = sales.loc[~sales.date.isin(pr_available[pr_available == 0].index),] # drop days without information
print('sales:', sales.order_datetime.min(), sales.order_datetime.max())

names = sales[['product_id','product_name']].groupby('product_id').first() # names of products
df = sales.merge(stock, how = 'inner', on = ['product_id','date'], validate = "m:1") # join df


result = stockout_functions.get_list_result(stock,sales,df, idpy = 67, idpx = 50)
pd.DataFrame(result)




