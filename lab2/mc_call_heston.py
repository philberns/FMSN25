import numpy as np

def mc_call_heston(S0, K, r, T, t, N, v0, kappa, theta, sigma_v, rho, n_steps=100):
    """
    Monte Carlo pricing of a European call option using the Heston stochastic volatility model.
    
    The Heston model:
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
    dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW2_t
    where dW1_t * dW2_t = rho * dt

    Parameters:
    S0 : float
        Initial stock price.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualized).
    T : float
        Maturity time of the option (in years).
    t : float
        Current time (in years).
    N : int
        Number of Monte Carlo simulations.
    v0 : float
        Initial variance (volatility squared).
    kappa : float
        Rate of mean reversion of variance.
    theta : float
        Long-term variance.
    sigma_v : float
        Volatility of variance (vol of vol).
    rho : float
        Correlation between stock and variance Brownian motions.
    n_steps : int
        Number of time steps for discretization.

    Returns:
    price : float
        Estimated price of the European call option.
    error : float
        Standard error of the estimate.
    """
    
    tau = T - t
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Precompute correlation structure
    sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)
    
    payoffs = []
    
    for i in range(N):
        # Initialize paths
        S = S0
        v = v0
        
        # Generate correlated random numbers for the entire path
        Z1 = np.random.randn(n_steps)
        Z2 = np.random.randn(n_steps)
        
        # Make Z2 correlated with Z1
        W1 = Z1
        W2 = rho * Z1 + sqrt_1_minus_rho2 * Z2
        
        # Simulate the path using Euler-Maruyama scheme
        for j in range(n_steps):
            # Ensure variance stays positive (Feller condition)
            v_sqrt = np.sqrt(np.maximum(v, 0))
            
            # Update variance process (CIR process)
            dv = kappa * (theta - v) * dt + sigma_v * v_sqrt * W2[j] * sqrt_dt
            v = v + dv
            
            # Update stock price process
            dS = r * S * dt + v_sqrt * S * W1[j] * sqrt_dt
            S = S + dS
        
        # Calculate payoff
        payoff = np.maximum(S - K, 0)
        payoffs.append(payoff)
    
    payoffs = np.array(payoffs)
    
    # Calculate price and error
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(N)
    
    return price, error


def mc_call_heston_mixed(S0, K, r, T, t, N, v0, kappa, theta, sigma_v, rho, n_steps=100):
    """
    Monte Carlo pricing using Milstein scheme for variance and Euler on log-scale for stock price.
    This combination is numerically stable and accurate.
    """
    
    tau = T - t
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Precompute correlation structure
    sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)
    
    payoffs = []
    
    for i in range(N):
        # Initialize paths
        log_S = np.log(S0)  # Work with log(S) for stability
        v = v0
        
        # Generate correlated random numbers for the entire path
        Z1 = np.random.randn(n_steps)
        Z2 = np.random.randn(n_steps)
        
        # Make Z2 correlated with Z1
        W1 = Z1
        W2 = rho * Z1 + sqrt_1_minus_rho2 * Z2
        
        # Simulate the path using mixed scheme
        for j in range(n_steps):
            # Ensure variance stays positive
            v = np.maximum(v, 0)
            v_sqrt = np.sqrt(v)
            
            # Milstein scheme for variance (more accurate for CIR process)
            dW2 = W2[j] * sqrt_dt
            dv = (kappa * (theta - v) * dt + 
                  sigma_v * v_sqrt * dW2 + 
                  0.25 * sigma_v**2 * (dW2**2 - dt))
            v = v + dv
            
            # Euler scheme for log(S) - ensures S stays positive
            dW1 = W1[j] * sqrt_dt
            d_log_S = (r - 0.5 * v) * dt + v_sqrt * dW1
            log_S = log_S + d_log_S
        
        # Convert back to stock price
        S_final = np.exp(log_S)
        
        # Calculate payoff
        payoff = np.maximum(S_final - K, 0)
        payoffs.append(payoff)
    
    payoffs = np.array(payoffs)
    
    # Calculate price and error
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(N)
    
    return price, error


def mc_call_heston_milstein(S0, K, r, T, t, N, v0, kappa, theta, sigma_v, rho, n_steps=100):
    """
    Monte Carlo pricing using Milstein scheme for both processes (kept for comparison).
    """
    
    tau = T - t
    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Precompute correlation structure
    sqrt_1_minus_rho2 = np.sqrt(1 - rho**2)
    
    payoffs = []
    
    for i in range(N):
        # Initialize paths
        S = S0
        v = v0
        
        # Generate correlated random numbers for the entire path
        Z1 = np.random.randn(n_steps)
        Z2 = np.random.randn(n_steps)
        
        # Make Z2 correlated with Z1
        W1 = Z1
        W2 = rho * Z1 + sqrt_1_minus_rho2 * Z2
        
        # Simulate the path using Milstein scheme
        for j in range(n_steps):
            # Ensure variance stays positive
            v = np.maximum(v, 0)
            v_sqrt = np.sqrt(v)
            
            # Milstein scheme for variance (more accurate for CIR process)
            dW2 = W2[j] * sqrt_dt
            dv = (kappa * (theta - v) * dt + 
                  sigma_v * v_sqrt * dW2 + 
                  0.25 * sigma_v**2 * (dW2**2 - dt))
            v = v + dv
            
            # Milstein scheme for stock price
            dW1 = W1[j] * sqrt_dt
            dS = (r * S * dt + 
                  v_sqrt * S * dW1 + 
                  0.5 * v * S * (dW1**2 - dt))
            S = S + dS
        
        # Calculate payoff
        payoff = np.maximum(S - K, 0)
        payoffs.append(payoff)
    
    payoffs = np.array(payoffs)
    
    # Calculate price and error
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(N)
    
    return price, error