import numpy as np

def mc_arithmetic_basket_antithetic(S0, K, r, sigma, t, T, N, iter, rho):
    tau = T - t
    # Construct correlation matrix
    rhomat = np.full([N, N], rho)
    np.fill_diagonal(rhomat, 1)
    
    # Covariance matrix for log(S): Σ = σσ'ρτ (without the *t factor)
    sigma_matrix = np.diag([sigma] * N) @ rhomat @ np.diag([sigma] * N) * tau
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(sigma_matrix)
    
    M = iter // 2
    payoffs = []
    for i in range(M):
        Z = np.random.randn(N)
        # Generate correlated random variables
        corr_Z = L @ Z
        
        # Stock prices with proper correlation structure
        S1 = S0 * np.exp((r - 0.5 * np.diag(sigma_matrix) / tau) * tau + corr_Z)
        S2 = S0 * np.exp((r - 0.5 * np.diag(sigma_matrix) / tau) * tau - corr_Z)
        
        payoff1 = np.maximum(np.mean(S1) - K, 0)
        payoff2 = np.maximum(np.mean(S2) - K, 0)
        payoffs.append((payoff1 + payoff2) / 2)
    
    payoffs = np.array(payoffs)
    price = np.exp(-r * tau) * np.mean(payoffs)
    error = np.exp(-r * tau) * np.std(payoffs) / np.sqrt(M)  # Note: M iterations, not iter
    
    return price, error
