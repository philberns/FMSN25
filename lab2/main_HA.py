import numpy as np
from scipy.stats import norm

from geometric_basket_call import geometric_basket_call
from mc_arithmetic_basket_control import mc_arithmetic_basket_control
K = [80, 100, 120]
T = 4
S0 = np.ones(12)*100
r = 0.02
sigma = 0.4
rho = 0.6
N = 12
t = 0
iter = 100000
def black_scholes_call(S, K, T, t, r, sigma):
    tau = T-t 
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return call_price

for i in K:
    price_geo = geometric_basket_call(S0, i, r, sigma, t, T, N, rho)
    price_bs = black_scholes_call(np.prod(S0)**(1/N), i, T, t, r, sigma)
    price_control, error_control = mc_arithmetic_basket_control(S0, i, r, sigma, t, T, N, iter, rho)
    print("Geometric Basket Option Price with K=", i, " Price: ", price_geo)
    print("Black-Scholes Option Price with K=", i, " Price: ", price_bs)
    print("Control Variates Basket Option Price with K=", i, " Price: ", price_control, " Error: ", error_control)