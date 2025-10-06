import numpy as np  
def mc_call_antithetic(S, K, r, sigma, t, T, N): 
    """
    Antithetic variates Monte Carlo pricing of a European call
    N = number of samples (will be halved internally, since we use pairs)
    """
    tau = T - t
    mu = (r - 0.5 * sigma**2) * tau
    vol = sigma * np.sqrt(tau)

    # Draw half as many normals, then use +/- pairs
    M = N // 2
    Z = np.random.randn(M)

    # Terminal stock prices for Z and -Z
    S_T1 = S * np.exp(mu + vol * Z)
    S_T2 = S * np.exp(mu + vol * (-Z))

    # Payoffs for both
    payoffs1 = np.maximum(S_T1 - K, 0)
    payoffs2 = np.maximum(S_T2 - K, 0)

    # Antithetic payoff = average of the pair
    payoffs = 0.5 * (payoffs1 + payoffs2)

    # Discounted mean and standard error
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(M)

    return price, error 
    
