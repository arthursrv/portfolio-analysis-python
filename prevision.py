from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Features techniques sur le S&P500
sp500 = data["^GSPC"].to_frame()
sp500_returns = sp500.pct_change().dropna()

# Indicateurs techniques
sp500["SMA_20"] = sp500["^GSPC"].rolling(20).mean()
sp500["SMA_50"] = sp500["^GSPC"].rolling(50).mean()
sp500["Volatility_20"] = sp500_returns["^GSPC"].rolling(20).std()
sp500["Return_1d"] = sp500_returns["^GSPC"]
sp500["Return_5d"] = sp500["^GSPC"].pct_change(5)
sp500["Return_20d"] = sp500["^GSPC"].pct_change(20)

# RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

sp500["RSI"] = compute_rsi(sp500["^GSPC"])

# Target : 1 si le marché monte demain, 0 sinon
sp500["Target"] = (sp500["^GSPC"].shift(-1) > sp500["^GSPC"]).astype(int)
sp500 = sp500.dropna()

# Train/Test split
features = ["SMA_20", "SMA_50", "Volatility_20", 
            "Return_1d", "Return_5d", "Return_20d", "RSI"]
X = sp500[features]
y = sp500["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# Modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print(f"\n🎯 Accuracy: {model.score(X_test, y_test):.2%}")
print(classification_report(y_test, model.predict(X_test)))

# Feature importance
importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind="barh", title="Importance des variables")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
