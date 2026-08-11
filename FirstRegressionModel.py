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
print(adftest) # p-value de 0.0432 por lo que es estacionario con un 95% de confianza 

# Ahora necesitamos hacer el half-life test para determinar el tiemp de mean-reversion
y=spread.diff().dropna() # calcula la diferencia del spread 
#dropna() es para eliminar los valores nulos (en este es el primero ya que no hay cambio)
z=sm.add_constant(spread.shift(1).dropna()) #mismo proceso de constante pero con el valor nuevo 
hfmodel=sm.OLS(y, z).fit() # OLS con el spread y su diferencia 
hfhedge_ratio=hfmodel.params[z.columns[1]]
print("hf:",hfhedge_ratio)


half_life = -np.log(2) /hfhedge_ratio # calculo del half-life con la formula 
# es la misma formula que se usa para calcular radioactive decay 
print("Half-life of mean reversion: ", half_life)

#Ahora vamos a usar un rolling window para calcular el rolling z-score 
#voy a usar un window de 120 dias (half-life of mean reversion) 

rollingmean=spread.rolling(window=120).mean() #rolling mean del spread
rollingstd=spread.rolling(window=120).std()  #STANDARD DEVIATION DEL SPREAD EN EL WINDOW 

zscore=(spread-rollingmean)/rollingstd.dropna() # formula ggez
plot=zscore.plot(figsize=(12,6), title="Z-SCORE OF SPREAD") 
plt.axhline(y=-1.5, color='r', linestyle=':', label='buy') 
plt.axhline(y=1.5, color='g', linestyle=':', label='sell') 
plt.legend()
plt.show()
print("z-score",zscore)

#Una vez tengo el z-score, podria hacer el backtest de la estrategia. Sin embargo, prefiero 
#hacer una division en periodos para verificar la estacionaridad el spread y la relacion de 
#cointegracion 

