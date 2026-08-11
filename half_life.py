#This function calculates the half-life of mean reversion for a given spread between two tickers. 
#As a prerequisite, the spread must be defined either manually or through the function 
#calc_hedge_ratio.

def half_life(spread)

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