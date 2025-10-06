import numpy as np
from scipy.stats import norm
from mc_call_antithetic import mc_call_antithetic
from mc_call_control import mc_call_controlvar
from mc_call_crude import mc_call_crude
from mc_arithmetic_basket_crude import mc_arithmetic_basket_crude
from mc_arithmetic_basket_antithetic import mc_arithmetic_basket_antithetic 
from mc_arithmetic_basket_control import mc_arithmetic_basket_control
from mc_call_heston import mc_call_heston_mixed
from mc_call_opout import mc_call_heston_upout
# == Parameters ==


# == black-scholes formula for comparison ==
def black_scholes_call(S, K, T, t, r, sigma):
    tau = T-t 
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return call_price

K = 80
r = 0.01
S0 = 90
sigma = 0.6 

pricebs= black_scholes_call(S0, K, 1, 0, r, sigma)
priceanti, erroranti=mc_call_antithetic(S0, K, r, sigma, 0, 1, 100000)
pricecontrol, errorcontrol=mc_call_controlvar(S0, K, r, sigma, 0, 1, 100000)
pricecrude, errorcrude=mc_call_crude(S0, K, r, sigma, 0, 1, 100000)

print("Black-Scholes Price: ", pricebs)
print("Crude Monte Carlo Price: ", pricecrude, " Error: ", errorcrude)
print("Antithetic Variates Price: ", priceanti, " Error: ", erroranti)
print("Control Variates Price: ", pricecontrol, " Error: ", errorcontrol)

K = [80, 100, 120]
r = 0.02
N = 12
S0 = np.ones(N)*100
rho = 0.6
sigma = 0.4 
t = 0
T= 1
iter = [1000, 10000, 100000]
print("\nBasket Option Prices:\n")
for i in K:
    for j in iter:
        pricecrude, errorcrude = mc_arithmetic_basket_crude(S0, i, r, sigma, t, T, N, j, rho)
        priceanti, erroranti = mc_arithmetic_basket_antithetic(S0, i, r, sigma, t, T, N, j, rho)
        pricecontrol, errorcontrol = mc_arithmetic_basket_control(S0, i, r, sigma, t, T, N, j, rho)
        print ("Crude Basket Option Price with K=", i, " and iterations ", j, " Price: ", pricecrude, " Error: ", errorcrude)
        print ("Antithetic Basket Option Price with K=", i, " and iterations ", j, " Price: ", priceanti, " Error: ", erroranti)
        print ("Control Variates Basket Option Price with K=", i, " and iterations ", j, " Price: ", pricecontrol, " Error: ", errorcontrol)

# == Heston Model Examples ==
print("\n" + "="*50)
print("HESTON STOCHASTIC VOLATILITY MODEL")
print("="*50)

# Heston model parameters
S0_heston = 90     # Initial stock price  
K_heston = 80      # Strike price
r_heston = 0.01     # Risk-free rate
T_heston = 1.0      # Time to maturity
t_heston = 0.0      # Current time
N_heston = 50000    # Number of simulations

# Heston-specific parameters
v0 = 0.16           # Initial variance (volatility^2)
kappa = 10.0         # Mean reversion speed
theta = 0.16        # Long-term variance
sigma_v = 0.1       # Volatility of variance
rho = -0.8          # Correlation between price and variance

print(f"\nHeston Model Parameters:")
print(f"S0={S0_heston}, K={K_heston}, r={r_heston}, T={T_heston}")
print(f"v0={v0}, kappa={kappa}, theta={theta}, sigma_v={sigma_v}, rho={rho}")
print(f"Number of simulations: {N_heston}")

# Compare with Black-Scholes (using initial volatility)
initial_vol = np.sqrt(v0)
bs_price = black_scholes_call(S0_heston, K_heston, T_heston, t_heston, r_heston, initial_vol)
print(f"\nBlack-Scholes (σ={initial_vol:.3f}): {bs_price:.4f}")


# Heston model with mixed scheme (Milstein for V, Euler log-scale for S)
heston_price_mixed, heston_error_mixed = mc_call_heston_mixed(
    S0_heston, K_heston, r_heston, T_heston, t_heston, N_heston,
    v0, kappa, theta, sigma_v, rho, n_steps=100
)
print(f"Heston (Mixed):       {heston_price_mixed:.4f} ± {heston_error_mixed:.4f}")
S0_upout = 50
B_upout = 100
K_upout = 40
# Heston model with mixed scheme (Milstein for V, Euler log-scale for S)
heston_price_upout, heston_error_upout = mc_call_heston_upout(
    S0_upout, K_upout, r_heston, T_heston, t_heston, N_heston,
    v0, kappa, theta, sigma_v, rho, B_upout, n_steps=100
)
print(f"Heston up-out (Mixed):       {heston_price_upout:.4f} ± {heston_error_upout:.4f}")
