
import math
import numpy as np
def european_call_binomial(S0, K, r, T, N, sigma):
    delta = T/N
    d= 0.5
    u =1.5
    #u = np.exp(sigma*np.sqrt(delta))
    #d = np.exp(-sigma*np.sqrt(delta))
    disc = math.exp(-r * delta)
    #p = (math.exp(r * delta) - d) / (u - d)
    qu = (np.exp(-r * T) -d)/(u - d)
    qd = (u - np.exp(-r * T))/(u - d)
    # --- Step 1: build stock price tree ---
    stock_tree = np.zeros((N+1,N+1))
    for i in range(N + 1):
        for j in range(i + 1):
            price = S0 * (u ** (j)) * (d ** (i-j))
            stock_tree[j][i]=(price)
    print(stock_tree)
    
    
    # --- Step 2: initialize option values at maturity ---
    payoff_tree = np.zeros((N+1,N+1))
    terminal_prices = stock_tree[:,-1]
    terminal_values = [max(price - K, 0) for price in terminal_prices]
    payoff_tree[:,-1] = terminal_values
    print(payoff_tree)
    for i in range(N-1,-1,-1):
        for j in range(i,-1,-1):
            payoff_tree[j, i]=disc*(qu*payoff_tree[j+1,i+1]+qd*payoff_tree[j,i+1])
    print(payoff_tree)

    rnvf = np.power(disc,3)*(np.power(qu,3)*terminal_values[0] + 3*np.power(qd,2)*qu*terminal_values[1] + 3*np.power(qu,2)*qd*terminal_values[2] + np.power(qu,3)*terminal_values[3])
    print(rnvf)
    
    # --- Step 3: roll back the tree ---+
    '''
    for i in range(N - 1, -1, -1):
        next_values = []
        for j in range(i + 1):
            expected = p * option_tree[0][j] + (1 - p) * option_tree[0][j + 1]
            value = disc * expected
            next_values.append(value)
        option_tree.insert(0, next_values)
    '''
    #return option_tree[0][0]  # option value today


# Example usage:
S0 = 80   # initial stock price
K = 80    # strike price
r = 0   # risk-free rate
T = 1      # time to maturity (1 year)
u = 1.1    # up factor
d = 0.9    # down factor
N = 3      # steps
sigma = 0.2
price = european_call_binomial(S0, K, r, T, N, sigma)
print(f"European call price: {price:.4f}")
