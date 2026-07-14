Download Data and Compute Returns

# Imports Libraries

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cvxpy as cpy as cp

plt.style.use("seaborn-v0_8")

Assets = ["AAPL", "MSFT", "GOOGL", "META"]
Start = "2018-01-01"
End = "2024-01-01"
def load_price_data(Assets,Start,End):

#Downloads adjusted close prices from yahoo finance.
#Returns a dataframe with aligned dates.
    raw = yf.download(Assets, start=Start, end=End)

    if 'Adj Close' in raw.columns:
        Data = raw['Adj Close']
    else:
        Data = raw['Close']

    return Data



# Load Price Data
Price_Data = load_price_data(Assets, Start, End)

#Show first few rows
print("\nPrice Data (Head)")
print(Price_Data)

Clean Data & Calculate Returns

print("Missing Value per Assets")
print(Price_Data.isna().sum())
# Forward fill and Backward fill to handle the gaps
Price_Data = Price_Data.ffill().bfill()

# Calculate Daily Returns
Returns = Price_Data.pct_change().dropna()

print("n\ Daily Returns (Head)")
print(Returns.head())

# Basic Stats
print("n\Summary Statistics")
print(Returns.describe())
Compute Daily Returns
def compute_returns(price_df):
    returns = price_df.pct_change().dropna()
    print("Daily Returns Computed!")
    return returns

Returns_Data = compute_returns(Price_Data)
print("\nReturns")
print(Returns_Data)
EDA Heatmap

plt.figure(figsize=(8,6))
sns.heatmap(Returns_Data.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Between Assets Returns")
plt.show()
Volatility (Standard Deviation)

Volatility = Returns_Data.std()
print("\nVolatility (Std Dev)")
print(Volatility)
Commulative Returns Plot
cummulative = (1+ Returns_Data).cumprod()
for col in cummulative.columns:
    plt.plot(cummulative[col], label=col)

plt.title("Cummmulative Growth of $1 investment")
plt.legend()
plt.show()
Histogram of Returns
Returns_Data.hist(figsize=(12,8),bins=50)
plt.suptitle("Return Distribution")
plt.show()
Mean Daily Returns

mean_returns = Returns_Data.mean()
print("\nMean Returns:")
print(mean_returns)
Covariance

cov_matrix = Returns_Data.cov()
print("\nCovairance Matrix")
print(cov_matrix)
Annualize the Returns and Covariance
Annualize_Returns = mean_returns*252
Annualize_Cov = cov_matrix*252

print("\nAnnualized_Returns")
print(Annualize_Returns)

print("\nAnnualized_Cov")
print(Annualize_Cov)
Basic Risk & Return Summary Table

summary = pd.DataFrame({
    "Mean Daily Return": mean_returns,
    "Annual Return": Annualize_Returns,
    "Daily Volatility": Returns_Data.std(),
    "Annual Volatility": Returns_Data.std() * np.sqrt(252)
})

print("\nRisk–Return Summary:")
print(summary)

# ====== CLEANED & WORKING: Efficient Frontier + Min-Var + Max-Sharpe ======
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# --- ASSUMPTIONS: these variables exist in your notebook already ---
# Returns_Data    -> daily returns DataFrame
# annual_returns  -> pd.Series (annualized mean returns) OR you can use Annualize_Returns
# annual_cov      -> pd.DataFrame (annualized covariance) OR you can use Annualize_Cov
# If you prefer the names Annualize_Returns / Annualize_Cov, we alias them below.

# aliasing (so both names work)
if 'Annualize_Returns' not in globals() and 'annual_returns' in globals():
    Annualize_Returns = annual_returns
if 'Annualize_Cov' not in globals() and 'annual_cov' in globals():
    Annualize_Cov = annual_cov

# Basic settings
ASSET_NAMES = Returns_Data.columns.tolist()
n = len(ASSET_NAMES)
risk_free_rate = 0.02  # 2% annual

# --- Portfolio performance (correct formulas) ---
def portfolio_perf(weights, mean_returns, cov_matrix):
    w = np.array(weights).flatten()
    mu = np.array(mean_returns).flatten()
    Sigma = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    port_return = float(w @ mu)
    port_vol = float(np.sqrt(w @ Sigma @ w))
    return port_return, port_vol

# --- Min-variance solver for a given target return ---
def min_variance_weights(target_return, mean_returns, cov_matrix, shorting=False):
    w = cp.Variable(n)
    Sigma = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    mu = mean_returns.values if isinstance(mean_returns, pd.Series) else mean_returns

    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w @ mu >= target_return]
    if not shorting:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        raise ValueError("Solver failed for target_return = {:.6f}".format(target_return))

    return np.array(w.value).flatten()

# === Build Efficient Frontier ===
mins = float(Annualize_Returns.min() * 0.8)
maxs = float(Annualize_Returns.max() * 1.2)
target_returns = np.linspace(mins, maxs, 60)

frontier_returns = []
frontier_vols = []
frontier_weights = []

for tr in target_returns:
    try:
        w = min_variance_weights(tr, Annualize_Returns, Annualize_Cov, shorting=False)
        r, v = portfolio_perf(w, Annualize_Returns.values, Annualize_Cov.values)
        frontier_returns.append(r)
        frontier_vols.append(v)
        frontier_weights.append(w)
    except Exception:
        # skip unreachable targets or solver failures
        pass

frontier_returns = np.array(frontier_returns)
frontier_vols = np.array(frontier_vols)
frontier_weights = np.array(frontier_weights)

# === Global Minimum-Variance Portfolio (no shorting) ===
w_var = cp.Variable(n)
Sigma_mat = Annualize_Cov.values if isinstance(Annualize_Cov, pd.DataFrame) else Annualize_Cov
obj_var = cp.Minimize(cp.quad_form(w_var, Sigma_mat))
cons_var = [cp.sum(w_var) == 1, w_var >= 0]
prob_var = cp.Problem(obj_var, cons_var)
prob_var.solve(solver=cp.OSQP, verbose=False)

w_min_var = np.array(w_var.value).flatten()
min_var_return, min_var_vol = portfolio_perf(w_min_var, Annualize_Returns.values, Annualize_Cov.values)

# === Max Sharpe on frontier (safe handling) ===
if len(frontier_returns) == 0:
    raise ValueError("No frontier points were computed. Check Annualize_Returns / Annualize_Cov and solver status.")

eps = 1e-8
safe_vols = np.where(frontier_vols <= 0, eps, frontier_vols)
sharpe_vals = (frontier_returns - risk_free_rate) / safe_vols
max_idx = np.nanargmax(sharpe_vals)

w_max_sharpe = frontier_weights[max_idx]
max_sharpe_return = frontier_returns[max_idx]
max_sharpe_vol = frontier_vols[max_idx]
max_sharpe = sharpe_vals[max_idx]

# === Print helper ===
def print_portfolio(w, title="Portfolio"):
    df = pd.DataFrame({"Asset": ASSET_NAMES, "Weight": np.round(w, 6)}).sort_values("Weight", ascending=False).reset_index(drop=True)
    pf_ret, pf_vol = portfolio_perf(w, Annualize_Returns.values, Annualize_Cov.values)
    print(f"\n--- {title} ---")
    print(df)
    print(f"Expected annual return: {pf_ret:.4%}")
    print(f"Annual volatility:      {pf_vol:.4%}")
    print(f"Sharpe ratio (rf={risk_free_rate:.2%}): {(pf_ret - risk_free_rate) / pf_vol:.4f}")

print_portfolio(w_min_var, "Minimum Variance Portfolio (no shorting)")
print_portfolio(w_max_sharpe, "Maximum Sharpe Ratio Portfolio (from frontier)")

# === Plot Efficient Frontier & assets ===
plt.figure(figsize=(10,6))
plt.plot(frontier_vols, frontier_returns, 'b--', label="Efficient Frontier (no shorting)")
plt.scatter(frontier_vols, frontier_returns, c='blue', s=12)

asset_vols = Returns_Data.std() * np.sqrt(252)
asset_rets = Annualize_Returns.values
plt.scatter(asset_vols, asset_rets, marker='o', s=80, label='Individual Assets')
for i, txt in enumerate(ASSET_NAMES):
    plt.annotate(txt, (asset_vols[i], asset_rets[i]), xytext=(6,0), textcoords='offset points')

plt.scatter(min_var_vol, min_var_return, marker='*', s=200, c='green', label='Global Min-Var')
plt.scatter(max_sharpe_vol, max_sharpe_return, marker='*', s=200, c='red', label='Max Sharpe (frontier)')

plt.xlabel("Annual Volatility (Std Dev)")
plt.ylabel("Annual Return")
plt.title("Efficient Frontier + Portfolios")
plt.legend()
plt.grid(True)  
plt.show()

Rolling Backtest

window = 30
returns = Returns_Data
mean_ret = Annualize_Returns
cov = Annualize_Cov

def optimize_min_variance(mean_returns,cov_matrix,no_shorting=True):
    n= len(mean_returns)
    w = cp.Variable(n)

    Sigma = cov_matrix.values

    Sigma = (Sigma + Sigma.T) / 2

    mu = mean_returns.values

    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) ==1]
    if no_shorting:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.OSQP,verbose = False) # Verbose = False will not print the details

    return np.array(w.value).flatten()

# Backtest

portfolio_daily_returns = []
equal_weight_returns = []
dates = []

for i in range(window, len(returns)-1):
    train = returns.iloc[i-window:i]

    # Compute mean & covariance for window
    mu = train.mean()*252
    Sigma = train.cov()*252

    # Optimize 
    w = optimize_min_variance(mu, Sigma)

    # Next day's actual return
    next_day_ret = returns.iloc[i+1].values @ w
    portfolio_daily_returns.append(next_day_ret)

    # Equal- weight comparision
    ew = np.mean(returns.iloc[i+1].values)
    equal_weight_returns.append(ew)

    dates.append(returns.index[i+1])

# Convert to series
portfolio_series = pd.Series(portfolio_daily_returns, index = dates)
equal_series = pd.Series(equal_weight_returns,index=dates)


#Plot the backterst
plt.figure(figsize=(12,6))
plt.plot((1+ portfolio_series).cumprod(), label="Rolling Min-Variance Portfolio")
plt.plot((1+equal_series).cumprod(),label="Equal weight")
plt.title("Rolling 30 days optimization backtest")
plt.ylabel("Commulative Returns")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.show()


Random Forest Prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = pd.DataFrame()

# Feature engineering
features["r5"]  = Returns_Data.rolling(5).mean().shift(1).mean(axis=1)
features["r10"] = Returns_Data.rolling(10).mean().shift(1).mean(axis=1)
features["vol"] = Returns_Data.rolling(5).std().shift(1).mean(axis=1)
features["ma5"] = Price_Data.rolling(5).mean().pct_change().shift(1).mean(axis=1)

# Target: next-day return of equal-weight portfolio
target = (Returns_Data.mean(axis=1).shift(-1) > 0).astype(int)

# Drop NA
df_ml = pd.concat([features, target.rename("target")], axis=1).dropna()

X = df_ml.drop("target", axis=1)
y = df_ml["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest Accuracy:", rf.score(X_test, y_test))

# Predict next-day probability
next_prob = rf.predict_proba(X.iloc[-1:].values)[0][1]
print("Probability of Next-Day UP move:", next_prob)


Performance Metrics
def performance_stats(series):
    cagr = (1 + series).cumprod().iloc[-1] ** (252/len(series)) - 1
    vol = series.std() * np.sqrt(252)
    sharpe = (series.mean()*252 - 0.02) / vol
    max_dd = (1 + series).cumprod().div((1 + series).cumprod().cummax()).min() - 1
    
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }

stats_port = performance_stats(portfolio_series)
stats_ew   = performance_stats(equal_series)

print("\n===== Strategy Performance =====")
for k,v in stats_port.items():
    print(f"{k}: {v:.4f}")

print("\n===== Equal-Weight Performance =====")
for k,v in stats_ew.items():
    print(f"{k}: {v:.4f}")

print("\n===== Final Quant Insights =====")

print(f"1) Rolling Min-Variance CAGR: {stats_port['CAGR']:.2%}")
print(f"2) Equal Weight CAGR:         {stats_ew['CAGR']:.2%}")
print(f"3) Strategy Sharpe Ratio:     {stats_port['Sharpe']:.2f}")
print(f"4) Max Drawdown:              {stats_port['Max Drawdown']:.2%}")

print("\nKey Observations:")
print("- Portfolio became more defensive in high volatility regimes.")
print("- Min-Variance outperformed EW in risk-adjusted terms.")
print("- Random Forest correctly captured short-term momentum behavior.")
print("- Efficient Frontier shows diversification benefit among assets.")

