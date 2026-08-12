#This function takes two tickers and a start date as input, pulls price data from 
#Yahoo Finance, calculates the hedge ratio using OLS regression, and plots the spread 
#between the log-prices of the two tickers.



def calc_hedge_ratio(t_1,t_2,sdate):


    data=yf.download([t_1,t_2], start=sdate, auto_adjust=True)["Close"] 
    #Descarga de precios 
    logdata=np.log(data) #Convierto los precios a logprices para medir el spread 
    #Aqui empiezo el modelo de regresion OLS 
    X=sm.add_constant(logdata[t_2])
    #logdata es un dataframe, entonces tomo como constante la columna t_2
    model=sm.OLS(logdata[t_1], X).fit()
    hedge_ratio=model.params[t_2]
    print(hedge_ratio)

    spread = logdata[t_1] - hedge_ratio * logdata[t_2]
    spread.plot(figsize=(12,6), title="Spread")
    plt.show()
    return spread 