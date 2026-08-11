#This function takes in the spread from calc_hedge_ratio and once the half-life and spread stationarity 
#have been determined using half_life, it calculates the rolling z-score using a set timeframe 


def z_score(spread, timeframe):
   rollingmean=spread.rolling(window=timeframe).mean() #rolling mean del spread
   rollingstd=spread.rolling(window=timeframe).std()  #STANDARD DEVIATION DEL SPREAD EN EL WINDOW 

   zscore=(spread-rollingmean)/rollingstd.dropna # formula ggez
   plot=zscore.plot(figsize=(12,6), title="Z-SCORE OF SPREAD") 
   plt.axhline(y=-1.5, color='r', linestyle=':', label='buy') 
   plt.axhline(y=1.5, color='g', linestyle=':', label='sell') 
   plt.legend()
  plt.show()
  print("z-score",zscore)