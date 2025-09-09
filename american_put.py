import numpy as np
import math
import european_call as euro
def american_put_derivate(S0, K, r, T, N, sigma):
    delta = T/N
    u = np.exp(sigma*np.sqrt(delta))
    d = np.exp(-sigma*np.sqrt(delta))
    disc = math.exp(-r * delta)
    qu = (np.exp(r * delta) -d)/(u - d)
    qd = (u - np.exp(r * delta))/(u - d)
    # --- Step 1: build stock price tree ---
    stock_tree = np.zeros((N+1,N+1))
    for i in range(N + 1):
        for j in range(i + 1):
            price = S0 * (u ** (j)) * (d ** (i-j))
            stock_tree[j][i]=(price)

# --- Step 2: initialize option values at maturity and step back in the payoff tree and determine whether the continual or exercise option is preferable ---
    payoff_tree = np.zeros((N+1,N+1))
    terminal_prices = stock_tree[:,-1]
    terminal_values = [max(K -price, 0) for price in terminal_prices]
    payoff_tree[:,-1] = terminal_values

    for i in range(N-1,-1,-1):
        for j in range(i,-1,-1):
            cont_value=disc*(qu*payoff_tree[j+1,i+1]+qd*payoff_tree[j,i+1])
            exercise_value = max(K - stock_tree[j,i],0)
            payoff_tree[j, i]=max(cont_value, exercise_value)            
    
    return payoff_tree[0,0]  # option value today

# Example usage:
S0 = 90   # initial stock price
K = 100    # strike price
r = 0.05   # risk-free rate
T = 1      # time to maturity (1 year)
N = [1,3,10]      # steps
sigma = 0.2
american_put_price = []
for i in N:
    american_put_price.append(american_put_derivate(S0, K, r, T, i, sigma))
    print(f"American put price with {i} steps: {american_put_derivate(S0, K, r, T, i, sigma):.4f}")
forward_price = S0 - K * math.exp(-r * T)
euro_call_price = euro.european_call_binomial(S0, K, r, T, 3, sigma)[0]
price_delta = euro_call_price-american_put_price[1]
print(f"Verify the put-call parity inequality at 0: {S0-K, price_delta,forward_price}")
  
