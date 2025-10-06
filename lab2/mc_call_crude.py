import numpy as np

def mc_call_crude(S, K, r, sigma, t, T, N):
    """
    Crude Monte Carlo pricing of a European call option.

    Parameters:
    S : float
        Current stock price.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock (annualized).
    t : float
        Current time (in years).
    T : float
        Maturity time of the option (in years).
    N : int
        Number of Monte Carlo simulations.

    Returns:
    price : float
        Estimated price of the European call option.
    error : float
        Standard error of the estimate.
    """
    

    tau = T - t
    mu = (r - 0.5 * sigma**2) * tau
    vol = sigma * np.sqrt(tau)
    zero = np.zeros(N)
    Z = np.random.randn(N)                   # standard normal samples
    S_T = S * np.exp(mu + vol * Z)  
    #print(np.size(S_T))        # terminal stock prices
    payoffs = np.maximum(S_T - K, zero)       # option payoffs

    price = np.exp(-r * tau) * np.mean(payoffs)  # discounted expectation
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(N)  # standard error

    return price, error
 
   