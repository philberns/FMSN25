import numpy as np



def mc_call_heston_upout(S0, K, r, T, t, N, v0, kappa, theta, sigma_v, rho, B, n_steps=100):
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
        if S_final < B:
            payoff = np.maximum(S_final - K, 0)
        else:
            payoff = 0
        payoffs.append(payoff)
    
    payoffs = np.array(payoffs)
    
    # Calculate price and error
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(N)
    
    return price, error


