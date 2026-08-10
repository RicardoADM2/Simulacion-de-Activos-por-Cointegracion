import yfinance as yf 
import numpy as np 
import statsmodels.api as sm 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stm


data=yf.download(["GLD","SLV"], start='2015-01-01', auto_adjust=True)["Close"] 
#Descarga de precios 
logdata=np.log(data) #Convierto los precios a logprices para medir el spread 
#Aqui empiezo el modelo de regresion OLS 
X=sm.add_constant(logdata["SLV"])
#logdata es un dataframe, entonces tomo como constante la columna GLD 
model=sm.OLS(logdata["GLD"], X).fit()
hedge_ratio=model.params["SLV"]
print(hedge_ratio)

spread = logdata["GLD"] - hedge_ratio * logdata["SLV"]
spread.plot(figsize=(12,6), title="Spread")
plt.show()

#Teniendo un hedge ratio de around 0.9 para el logprice, entendemos que por cada 1% que se 
#mueve SLV, GLD se mueve un 0.9%

#Ahora hacemos el test de adf

adftest=stm.adfuller(spread)
print(adftest)