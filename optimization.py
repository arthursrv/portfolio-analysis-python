from scipy.optimize import minimize

# Simulation 10 000 portefeuilles
num_portfolios = 10000
num_assets = len(tickers) - 1  # sans le S&P500
returns_clean = returns.drop("^GSPC", axis=1)

results = np.zeros((4, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)

    port_return = np.sum(returns_clean.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T,
               np.dot(returns_clean.cov() * 252, weights)))

    results[0,i] = port_return
    results[1,i] = port_vol
    results[2,i] = (port_return - rf) / port_vol  # Sharpe

# Portefeuille optimal (max Sharpe)
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]

print("\n🏆 Portefeuille Optimal (Max Sharpe):")
for ticker, weight in zip(returns_clean.columns, optimal_weights):
    print(f"  {ticker}: {weight:.1%}")
print(f"  Rendement: {results[0,max_sharpe_idx]:.2%}")
print(f"  Volatilité: {results[1,max_sharpe_idx]:.2%}")
print(f"  Sharpe: {results[2,max_sharpe_idx]:.2f}")

# Visualisation frontière efficiente
plt.figure(figsize=(12,8))
plt.scatter(results[1,:], results[0,:], c=results[2,:],
            cmap="viridis", alpha=0.5, s=10)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx],
            color="red", s=200, marker="*", label="Portefeuille Optimal")
plt.xlabel("Volatilité annualisée")
plt.ylabel("Rendement annualisé")
plt.title("Frontière Efficiente — Optimisation de Portefeuille")
plt.legend()
plt.savefig("frontiere_efficiente.png", dpi=150)
plt.show()