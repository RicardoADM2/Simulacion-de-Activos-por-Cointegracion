#Lo voy a estar explicando todo al principio para no perderme y practicar python
import yfinance as yf 
import matplotlib.pyplot as plt 
import numpy as np
#Para ambos esto es importar los libraries y asginarle nombres para usar 

data=yf.download(["GLD","SLV"], start='2015-01-01', auto_adjust=True)["Close"]
logdata=np.log(data)
#download(['']) me deja sacar los tickers, start='' me da el timeframe, autoadjust ajusta(duh)
print(logdata.head())
print(logdata.shape)

logdata.plot(figsize=(12,6))
#logprice es lo que es mas util al usar cointegracion ya que 
plt.ylabel('Price')
plt.show()