
# Récupération données
import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["AAPL", "MSFT", "GOOGL", "BNP.PA", "MC.PA",
           "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "^GSPC"]

data = yf.download(tickers, start="2020-01-01", end="2024-12-31")["Close"]
returns = data.pct_change().dropna()

print(data.tail())
print(returns.describe())
# Volatilité annualisée
volatility = returns.std() * np.sqrt(252)

# Rendement annualisé
annual_return = returns.mean() * 252

# Sharpe Ratio (taux sans risque = 3%)
rf = 0.03
sharpe = (annual_return - rf) / volatility

# Value at Risk 95%
VaR_95 = returns.quantile(0.05)

# Maximum Drawdown
def max_drawdown(prices):
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.min()

mdd = data.apply(max_drawdown)

# Tableau récapitulatif
summary = pd.DataFrame({
    "Rendement Annuel": annual_return,
    "Volatilité": volatility,
    "Sharpe Ratio": sharpe,
    "VaR 95%": VaR_95,
    "Max Drawdown": mdd
})
print(summary.round(3))