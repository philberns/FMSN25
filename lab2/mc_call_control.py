import numpy as np

def mc_call_controlvar(S, K, r, sigma, t, T, N):
    """
    Monte-Carlo with control variates for a European call option.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying stock
    t : float
        Current time
    T : float
        Maturity time
    N : int
        Number of Monte Carlo simulations

    Returns:
    price : float
        Estimated option price
    error : float
        Standard error of the estimate
    """
    tau = T - t
    mu = (r - 0.5 * sigma**2) * tau
    vol = sigma * np.sqrt(tau)

    Z = np.random.randn(N)
    S_T = S * np.exp(mu + vol * Z)
    payoffs = np.maximum(S_T - K, 0)

    # Control variate: underlying asset S_T
    Y = S_T                       # control variable
    EY = S * np.exp(r * tau)      # expected value of S_T

    # Estimate optimal c (regression)
    c = np.cov(payoffs, Y)[0, 1] / np.var(Y)

    # Adjusted payoff
    payoffs_cv = payoffs + c * (EY - Y)

    # Discounted mean and standard error
    price = np.exp(-r * tau) * np.mean(payoffs_cv)
    error = np.exp(-r * tau) * np.std(payoffs_cv) / np.sqrt(N)

    return price, error