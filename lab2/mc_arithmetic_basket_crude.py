import numpy as np 

def mc_arithmetic_basket_crude(S0, K, r, sigma, t, T, N, iter, rho):
    tau = T - t
    # Construct correlation matrix
    rhomat = np.full([N, N], rho)
    np.fill_diagonal(rhomat, 1)
    
    # Covariance matrix for log(S): Σ = σσ'ρτ
    sigma_matrix = np.diag([sigma] * N) @ rhomat @ np.diag([sigma] * N) * tau
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(sigma_matrix)
    
    payoffs = []  # Store all payoffs
    for i in range(iter):
        Z = np.random.randn(N)
        # Generate correlated random variables
        corr_Z = L @ Z
        
        # Stock prices with proper correlation structure
        S = S0 * np.exp((r - 0.5 * np.diag(sigma_matrix) / tau) * tau + corr_Z)
        payoff = np.maximum(np.mean(S) - K, 0)
        payoffs.append(payoff)
    
    # Calculate price and error after collecting all payoffs
    payoffs = np.array(payoffs)
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(iter)
     
    return price, error