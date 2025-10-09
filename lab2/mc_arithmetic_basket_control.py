import numpy as np 
from scipy.stats import norm
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
    # Match the calculation from geometric_basket_call function
    weights = np.ones(N) / N
    sigma_geom_squared_total = tau * sigma**2 * (np.sum(weights**2) + (1 - np.sum(weights**2)) * rho)
    
    
    # Geometric mean of initial prices
    S_geom = np.prod(S0)**(1/N)
    
    # Black-Scholes price for geometric basket (our control variate)
    # Match the calculation from geometric_basket_call function
    
    a = np.log(S_geom) + (r - 0.5*sigma**2)*tau
    b = sigma_geom_squared_total
    
    d1 = (a - np.log(K) + b) / np.sqrt(b)
    d2 = d1 - np.sqrt(b)
    control_theoretical = (np.exp(a + 0.5*b)*norm.cdf(d1) - K*norm.cdf(d2))
    
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
    print(f"Mean payoff (arithmetic): {np.mean(payoffs_arithmetic):.6f}")
    print(f"Mean payoff (geometric): {np.mean(payoffs_geometric):.6f}")
    # Control variate technique
    # Optimal control coefficient: β* = Cov(X,Y) / Var(Y)
    cov_xy = np.cov(payoffs_arithmetic, payoffs_geometric)[0, 1]
    var_y = np.var(payoffs_geometric)
    
    if var_y > 0:
        beta_optimal = cov_xy / var_y
        
    else:
        beta_optimal = 0
    
    # Control variate estimator: X̃ = X - β*(Y - E[Y])
    # Note: payoffs are undiscounted, so we need undiscounted theoretical value
    
    controlled_payoffs = payoffs_arithmetic - beta_optimal * (payoffs_geometric - control_theoretical)
    
    # Calculate price and error
    price = np.exp(-r * tau) * np.mean(controlled_payoffs)
    error = np.exp(-r * tau) * np.std(controlled_payoffs) / np.sqrt(iter)
    
    return price, error
