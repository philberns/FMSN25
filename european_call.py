#%%
import math
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as si
def european_call_binomial(S0, K, r, T, N, sigma):
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
    
    
    
    # --- Step 2: initialize option values at maturity and step back in the payoff tree ---
    payoff_tree = np.zeros((N+1,N+1))
    terminal_prices = stock_tree[:,-1]
    terminal_values = [max(price - K, 0) for price in terminal_prices]
    payoff_tree[:,-1] = terminal_values

    for i in range(N-1,-1,-1):
        for j in range(i,-1,-1):
            payoff_tree[j, i]=disc*(qu*payoff_tree[j+1,i+1]+qd*payoff_tree[j,i+1])


    rnvf = np.power(disc,N)*np.sum([math.comb(N, k) * (qu ** k) * (qd ** (N - k)) * terminal_values[k] for k in range(N + 1)])
    #rnvf = np.power(disc,N)*(np.power(qu,N)*terminal_values[0] + N*np.power(qd,N-1)*qu*terminal_values[1] + N*np.power(qu,N-1)*qd*terminal_values[N-1] + np.power(qu,N)*terminal_values[3])
    
    return payoff_tree[0,0], rnvf  # option value today


# Example usage:
S0 = 90   # initial stock price
K = 100    # strike price
r = 0.05   # risk-free rate
T = 1      # time to maturity (1 year)
N = 3      # steps
sigma = 0.2
price, rnvf_price = european_call_binomial(S0, K, r, T, N, sigma)
print(f"European call price: {price:.4f}, {rnvf_price:.4f}")

x = np.arange(1, 101)  # integers from 1 to 100
euro_prices_convergence = np.zeros(100)
for idx, N in enumerate(x):
    euro_prices_convergence[idx] = european_call_binomial(S0, K, r, T, N, sigma)[0]


def F(t,s):
    d1 = lambda t,s: (math.log(s/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*math.sqrt(T-t))
    d2 = lambda t,s: d1(t,s) - sigma*math.sqrt(T-t)
    return s*si.norm.cdf(d1(t,s)) - K*math.exp(-r*(T-t))*si.norm.cdf(d2(t,s))

plt.plot(x, euro_prices_convergence)
plt.axhline(F(0,S0), linestyle='--', color='red')
plt.xlabel('Number of Periods (N)')
plt.ylabel('European Call Price')
plt.title('Convergence of European Call Price with Increasing Periods')
plt.grid()
plt.show()  
# %%
