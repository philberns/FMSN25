
import math

def european_call_binomial(S0, K, r, T, u, d, N):
    dt = T / N
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)

    # --- Step 1: build stock price tree ---
    stock_tree = []
    for i in range(N + 1):
        level = []
        for j in range(i + 1):
            price = S0 * (u ** (i - j)) * (d ** j)
            level.append(price)
        stock_tree.append(level)

    # --- Step 2: initialize option values at maturity ---
    option_tree = []
    terminal_prices = stock_tree[-1]
    terminal_values = [max(price - K, 0) for price in terminal_prices]
    option_tree.append(terminal_values)

    # --- Step 3: roll back the tree ---
    for i in range(N - 1, -1, -1):
        next_values = []
        for j in range(i + 1):
            expected = p * option_tree[0][j] + (1 - p) * option_tree[0][j + 1]
            value = disc * expected
            next_values.append(value)
        option_tree.insert(0, next_values)

    return option_tree[0][0]  # option value today


# Example usage:
S0 = 100   # initial stock price
K = 100    # strike price
r = 0.05   # risk-free rate
T = 1      # time to maturity (1 year)
u = 1.1    # up factor
d = 0.9    # down factor
N = 3      # steps

price = european_call_binomial(S0, K, r, T, u, d, N)
print(f"European call price: {price:.4f}")
