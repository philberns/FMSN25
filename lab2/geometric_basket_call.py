import numpy as np
from scipy.stats import norm
def geometric_basket_call(S0, K, r, sigma, t, T, N, rho):
    tau = T-t
    weights = 1/N*np.ones(N)
    S0 = np.atleast_1d(S0)  # Convert int/float to 1D array, leave arrays unchanged
    a = np.log(np.mean(S0)) + (r-0.5*sigma**2)*tau 
    b = T*sigma**2*(np.sum(weights**2) + (1 - np.sum(weights**2) )*rho )

    d1 = (a - np.log(K) + b)/(np.sqrt(b))
    d2 = d1 - np.sqrt(b)
    price = np.exp(-r*tau)*(np.exp(a + 0.5*b)*norm.cdf(d1) - K*norm.cdf(d2))

    return price
