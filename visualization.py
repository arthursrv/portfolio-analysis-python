import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Evolution des prix normalisés
(data / data.iloc[0] * 100).plot(ax=axes[0,0], title="Performance normalisée (base 100)")

# Schéma corrélations
import seaborn as sns
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm",
            ax=axes[0,1], fmt=".2f")
axes[0,1].set_title("Matrice de corrélation")

# Distribution des rendements
returns["AAPL"].hist(bins=50, ax=axes[1,0], color="steelblue")
axes[1,0].set_title("Distribution des rendements AAPL")
axes[1,0].axvline(VaR_95["AAPL"], color="red", label=f"VaR 95%: {VaR_95['AAPL']:.2%}")
axes[1,0].legend()

# Sharpe Ratio comparatif
sharpe.plot(kind="bar", ax=axes[1,1], color="steelblue", title="Sharpe Ratio par action")

plt.tight_layout()
plt.savefig("analyse_portefeuille.png", dpi=150)
plt.show()