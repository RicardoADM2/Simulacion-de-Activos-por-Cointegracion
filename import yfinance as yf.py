import yfinance as yf
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stm

# ==========================================================
# 1. Data & spread construction (unchanged from your code)
# ==========================================================
data = yf.download(["GLD", "SLV"], start='2015-01-01', auto_adjust=True)["Close"]
logdata = np.log(data)

X = sm.add_constant(logdata["SLV"])
model = sm.OLS(logdata["GLD"], X).fit()
hedge_ratio = model.params["SLV"]
print("Hedge ratio:", hedge_ratio)

spread = logdata["GLD"] - hedge_ratio * logdata["SLV"]

adftest = stm.adfuller(spread)
print("ADF p-value:", adftest[1])

y = spread.diff().dropna()
z = sm.add_constant(spread.shift(1).dropna())
hfmodel = sm.OLS(y, z).fit()
hfhedge_ratio = hfmodel.params[z.columns[1]]
half_life = -np.log(2) / hfhedge_ratio
print("Half-life of mean reversion:", half_life)

window = 120  # you were using the half-life as the rolling window
rollingmean = spread.rolling(window=window).mean()
rollingstd = spread.rolling(window=window).std()
zscore = (spread - rollingmean) / rollingstd

# ==========================================================
# 2. Turn the buy/sell z-score levels into an actual position
# ==========================================================
# Rule: enter when |z| > entry_z, flatten once z crosses back through exit_z.
# This matches the -1.5 / +1.5 lines you drew on the z-score plot.
entry_z = 1.5
exit_z = 0.0

position = pd.Series(index=zscore.index, dtype=float)
position[zscore < -entry_z] = 1     # spread unusually low -> long the spread (long GLD, short hedge*SLV)
position[zscore > entry_z] = -1     # spread unusually high -> short the spread
# flatten the trade the day the spread reverts through the mean
position[(zscore >= exit_z) & (zscore.shift(1) < exit_z)] = 0
position[(zscore <= exit_z) & (zscore.shift(1) > exit_z)] = 0

position = position.ffill().fillna(0)

# ==========================================================
# 3. Backtest, including transaction costs
# ==========================================================
# spread.diff() is ~ the log-return of a portfolio that is
# long 1 unit of GLD and short `hedge_ratio` units of SLV,
# so position.shift(1) * spread.diff() is that portfolio's
# daily strategy log-return before costs.
fee_bps = 5  # cost per leg per trade, in basis points of notional (5bps = 0.05%); change to match your broker
notional_per_unit = 1 + hedge_ratio  # 1 unit GLD leg + hedge_ratio units SLV leg

spread_ret = spread.diff()
trade = position.diff().abs().fillna(0)   # 0 = no change, 1 = open/close, 2 = flip long<->short
fees = trade * notional_per_unit * (fee_bps / 10000)

strategy_ret_gross = position.shift(1) * spread_ret
strategy_ret_net = (strategy_ret_gross - fees).dropna()
strategy_ret_gross = strategy_ret_gross.dropna()

equity_gross = np.exp(strategy_ret_gross.cumsum())
equity_net = np.exp(strategy_ret_net.cumsum())

# ==========================================================
# 4. Performance stats
# ==========================================================
def perf_stats(ret, label):
    ann_factor = 252
    total_return = np.exp(ret.sum()) - 1
    n_years = len(ret) / ann_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = ret.std() * np.sqrt(ann_factor)
    sharpe = (ret.mean() * ann_factor) / ann_vol if ann_vol > 0 else np.nan
    cum = np.exp(ret.cumsum())
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    print(f"--- {label} ---")
    print(f"Total return:   {total_return:.2%}")
    print(f"CAGR:           {cagr:.2%}")
    print(f"Annualized vol: {ann_vol:.2%}")
    print(f"Sharpe ratio:   {sharpe:.2f}")
    print(f"Max drawdown:   {max_dd:.2%}")
    print()

perf_stats(strategy_ret_gross, "Gross (no fees)")
perf_stats(strategy_ret_net, "Net (with fees)")

n_trades = int((trade > 0).sum())
print(f"Number of position changes (trades): {n_trades}")
print(f"Cumulative fee drag (log-return units): {fees.sum():.4f}")
print(f"Fee drag on total return: {(equity_gross.iloc[-1] - equity_net.iloc[-1]) / equity_gross.iloc[-1]:.2%}")

# ==========================================================
# 5. Plots
# ==========================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(zscore.index, zscore, label="Z-score")
axes[0].axhline(entry_z, color='r', linestyle=':', label='sell (short spread)')
axes[0].axhline(-entry_z, color='r', linestyle='-', label='buy (long spread)')
axes[0].axhline(exit_z, color='k', linestyle='--', linewidth=0.8, label='exit')
axes[0].set_title("Z-score of spread with entry/exit thresholds")
axes[0].legend(loc="upper left", fontsize=8)

axes[1].plot(position.index, position, drawstyle='steps-post')
axes[1].set_title("Position (+1 long spread, -1 short spread, 0 flat)")
axes[1].set_ylim(-1.5, 1.5)

axes[2].plot(equity_gross.index, equity_gross, label="Gross equity (no fees)")
axes[2].plot(equity_net.index, equity_net, label=f"Net equity ({fee_bps}bps/leg/trade)")
axes[2].set_title("Equity curve (starting capital = 1)")
axes[2].legend()

plt.tight_layout()
plt.savefig("gld_slv_backtest_equity.png", dpi=150)
plt.show()