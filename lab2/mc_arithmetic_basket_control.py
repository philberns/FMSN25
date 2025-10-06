import numpy as np 

def mc_arithmetic_basket_control(S0, K, r, sigma, t, T, N, iter, rho):
    tau = T - t
    # Construct correlation matrix
    rhomat = np.full([N, N], rho)
    np.fill_diagonal(rhomat, 1)
    
    # Covariance matrix for log(S): Σ = σσ'ρτ
    sigma_matrix = np.diag([sigma] * N) @ rhomat @ np.diag([sigma] * N) * tau
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(sigma_matrix)
    
    # Calculate theoretical price of geometric basket option (control variate)
    # For geometric basket: σ_geom² = (1/N²) * 1ᵀ Σ 1 where 1 is vector of ones
    ones = np.ones(N)
    sigma_geom_squared = (1/N**2) * ones.T @ sigma_matrix @ ones / tau
    sigma_geom = np.sqrt(sigma_geom_squared)
    
    # Adjusted parameters for geometric basket
    S_geom = np.prod(S0)**(1/N)  # Geometric mean of initial prices
    r_adj = r - 0.5 * sigma_geom_squared + 0.5 * sigma**2  # Drift adjustment
    
    # Black-Scholes price for geometric basket (our control variate)
    from scipy.stats import norm
    d1 = (np.log(S_geom / K) + (r_adj + 0.5 * sigma_geom**2) * tau) / (sigma_geom * np.sqrt(tau))
    d2 = d1 - sigma_geom * np.sqrt(tau)
    control_theoretical = S_geom * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    
    payoffs_arithmetic = []
    payoffs_geometric = []
    
    for i in range(iter):
        Z = np.random.randn(N)
        # Generate correlated random variables
        corr_Z = L @ Z
        
        # Stock prices with proper correlation structure
        S = S0 * np.exp((r - 0.5 * np.diag(sigma_matrix) / tau) * tau + corr_Z)
        
        # Calculate payoffs
        payoff_arithmetic = np.maximum(np.mean(S) - K, 0)
        payoff_geometric = np.maximum(np.prod(S)**(1/N) - K, 0)
        
        payoffs_arithmetic.append(payoff_arithmetic)
        payoffs_geometric.append(payoff_geometric)
    
    payoffs_arithmetic = np.array(payoffs_arithmetic)
    payoffs_geometric = np.array(payoffs_geometric)
    
    # Control variate technique
    # Optimal control coefficient: β* = Cov(X,Y) / Var(Y)
    cov_xy = np.cov(payoffs_arithmetic, payoffs_geometric)[0, 1]
    var_y = np.var(payoffs_geometric)
    
    if var_y > 0:
        beta_optimal = cov_xy / var_y
    else:
        beta_optimal = 0
    
    # Control variate estimator: X̃ = X - β*(Y - E[Y])
    controlled_payoffs = payoffs_arithmetic - beta_optimal * (payoffs_geometric - control_theoretical)
    
    # Calculate price and error
    price = np.exp(-r * tau) * np.mean(controlled_payoffs)
    error = np.exp(-r * tau) * np.std(controlled_payoffs) / np.sqrt(iter)
    
    return price, error
