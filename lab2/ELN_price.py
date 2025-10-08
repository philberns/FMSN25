import numpy as np
from scipy.stats import norm

def mc_arithmetic_basket_control_custom_weights(S0, weights, K, r, sigma_array, t, T, correlation_matrix, n_iter, seed=None):
    tau = T - t
    N_assets = len(S0)

    # Recover c_i (geometric weights that sum to 1) if weights = c_i / S0
    c_weights = weights * S0
    
    # Precompute objects
    logS0 = np.log(S0)
    # full covariance without T factor
    Sigma = np.outer(sigma_array, sigma_array) * correlation_matrix

    # geometric log-variance and mean (for log G)
    sigma_geom_sq = c_weights @ (Sigma @ c_weights) * tau    # scalar
    mu_geom = np.sum(c_weights * (logS0 + (r - 0.5 * sigma_array**2) * tau))

    # analytic UNdiscounted E[(G_T - K)+]
    s = np.sqrt(sigma_geom_sq)
    
    d1 = (mu_geom + sigma_geom_sq - np.log(K)) / s
    d2 = (mu_geom - np.log(K)) / s
    control_theoretical = np.exp(mu_geom + 0.5*sigma_geom_sq) * norm.cdf(d1) - K * norm.cdf(d2)
    
    # prepare Cholesky of correlation once
    Lcorr = np.linalg.cholesky(correlation_matrix)
    payoffs_arith = np.empty(n_iter)
    payoffs_geom = np.empty(n_iter)

    for i in range(n_iter):
        Z = np.random.randn(N_assets)
        W = Lcorr @ Z                               # correlated standard normals
        S = S0 * np.exp((r - 0.5*sigma_array**2) * tau + sigma_array * np.sqrt(tau) * W)

        # arithmetic payoff (undiscounted)
        basket_value_arith = np.sum(weights * S)
        payoff_arith = max(basket_value_arith - K, 0.0)

        # geometric payoff with same c_weights (undiscounted)
        basket_value_geom = np.exp(np.sum(c_weights * np.log(S)))
        payoff_geom = max(basket_value_geom - K, 0.0)

        payoffs_arith[i] = payoff_arith
        payoffs_geom[i] = payoff_geom

    # sample cov/var and optimal beta (use unbiased cov if desired)
    cov_xy = np.cov(payoffs_arith, payoffs_geom, ddof=1)[0,1]
    var_y = np.var(payoffs_geom, ddof=1)
    beta_opt = cov_xy / var_y 

    # control-variate (all undiscounted)
    controlled = payoffs_arith - beta_opt * (payoffs_geom - control_theoretical)

    # discount once at the end
    price = np.exp(-r * tau) * np.mean(controlled)
    stderr = np.exp(-r * tau) * np.std(controlled, ddof=1) / np.sqrt(n_iter)
    
    return price, stderr

    
def eln_pricing_with_participation_rate(NA, c_i, S0, r, T, sigma_array, correlation_matrix, n_iter=100000, target_multiple=1.1):
    # Modified weights for basket
    weights = c_i / S0
    # Price the arithmetic basket call with K=1
    print(f"\nPricing arithmetic basket call with K={1}...")
    basket_call_price, basket_call_error = mc_arithmetic_basket_control_custom_weights(
        S0=S0, weights=weights, K=1.0, r=r, 
        sigma_array=sigma_array, t=0, T=T, 
        correlation_matrix=correlation_matrix, n_iter=n_iter
    )
    
    print(f"Arithmetic basket call price: {basket_call_price:.6f} ± {basket_call_error:.6f}")
    
    # Calculate required participation rate
    discount_factor = np.exp(-r * T)
    required_basket_component = target_multiple - discount_factor
    

    participation_rate = required_basket_component / basket_call_price
    
    print(f"Participation rate ρT: {participation_rate:.6f}")
        
        # Verify the ELN price
    eln_price = discount_factor * NA + NA * participation_rate * basket_call_price

    return {
        'participation_rate': participation_rate,
        'eln_price': eln_price
    }

def eln_pricing_at_time_t(NA, participation_rate, c_i, S0_original, S_current, r, t, T, sigma_array, correlation_matrix, n_iter=100000):
    
    tau = T - t  # Remaining time to maturity
    # Modified weights are based on ORIGINAL prices (fixed at inception)
    weights = c_i / S0_original

    
    
    # Simple approach: Price the ELN payoff from current state to maturity
    # We simulate B(T)/B(0) starting from current basket level
    # The payoff is max(B(T)/B(0) - 1, 0)
    
    current_basket_value = np.sum(weights * S_current)
    
    
    
    # Price basket call with strike K=1 (since we want max(B(T)/B(0) - 1, 0))
    # Starting from current prices, simulating to maturity
    
    basket_call_price, __ = mc_arithmetic_basket_control_custom_weights(
        S0=S_current, weights=weights, K=1.0, r=r,
        sigma_array=sigma_array, t=0, T=tau,  # Note: using tau as time to maturity
        correlation_matrix=correlation_matrix, n_iter=n_iter
    )
    
    # Calculate ELN components
    bond_component = np.exp(-r * tau) * NA
    basket_component = NA * participation_rate * basket_call_price
    eln_price_current = bond_component + basket_component
    
    return {
        'eln_price_at_t': eln_price_current
    }

# Example usage for t=1 pricing
if __name__ == "__main__":
    
    # ELN Test case with your data (using same parameters as first test)
    print("\nELN Pricing Test:")
    c_i = np.array([0.07, 0.11, 0.11, 0.18, 0.06, 0.04, 0.01, 0.17, 0.04, 0.02, 0.03, 0.16])
    S0 = np.array([141.3, 80.9, 193.3, 316.7, 212.9, 75.6, 236.6, 108.8, 20.68, 192.9, 46.55, 88.95])
    
    # Example volatilities and correlation matrix (you should use your actual data)
    sigma_array = np.array([0.32, 0.22, 0.31, 0.18, 0.27, 0.22, 0.21, 0.23, 0.34, 0.17, 0.18, 0.23])
    
    correlation_matrix = np.array([
        [1.00, 0.30, 0.28, 0.32, 0.36, 0.52, 0.29, 0.37, 0.44, 0.41, 0.38, 0.41],
        [0.30, 1.00, 0.17, 0.45, 0.44, 0.42, 0.37, 0.35, 0.32, 0.42, 0.42, 0.27],
        [0.28, 0.17, 1.00, 0.20, 0.17, 0.31, 0.23, 0.24, 0.20, 0.25, 0.27, 0.21],
        [0.32, 0.45, 0.20, 1.00, 0.34, 0.38, 0.39, 0.30, 0.32, 0.43, 0.42, 0.30],
        [0.36, 0.44, 0.17, 0.34, 1.00, 0.47, 0.22, 0.33, 0.39, 0.38, 0.31, 0.31],
        [0.52, 0.42, 0.31, 0.38, 0.47, 1.00, 0.37, 0.37, 0.46, 0.47, 0.42, 0.56],
        [0.29, 0.37, 0.23, 0.39, 0.22, 0.37, 1.00, 0.25, 0.26, 0.42, 0.29, 0.22],
        [0.37, 0.35, 0.24, 0.30, 0.33, 0.37, 0.25, 1.00, 0.39, 0.39, 0.34, 0.26],
        [0.44, 0.32, 0.20, 0.32, 0.39, 0.46, 0.26, 0.39, 1.00, 0.43, 0.38, 0.29],
        [0.41, 0.42, 0.25, 0.43, 0.38, 0.47, 0.42, 0.39, 0.43, 1.00, 0.46, 0.38],
        [0.38, 0.42, 0.27, 0.42, 0.31, 0.42, 0.29, 0.34, 0.38, 0.46, 1.00, 0.39],
        [0.41, 0.27, 0.21, 0.30, 0.31, 0.56, 0.22, 0.26, 0.29, 0.38, 0.39, 1.00]
    ])
    
    results = eln_pricing_with_participation_rate(
        NA=100, c_i=c_i, S0=S0, r=-0.001, T=5,
        sigma_array=sigma_array, correlation_matrix=correlation_matrix,
        n_iter=50000, target_multiple=1.1
    )
    
    
    print("ELN PRICING AFTER 1 YEAR (t=1)")

    # Actual stock prices at t=1 (provided data)
    S_current = np.array([179.3, 57.45, 163.7, 260.6, 220.4, 89.85, 259.5, 143.8, 20.83, 194.7, 37.62, 97.5])
    
    results_t1 = eln_pricing_at_time_t(
        NA=100,
        participation_rate=results['participation_rate'],  # Use the rate from initial pricing
        c_i=c_i,
        S0_original=S0,  # Original prices at t=0
        S_current=S_current,  # Current prices at t=1
        r=-0.001, t=1, T=5,
        sigma_array=sigma_array,
        correlation_matrix=correlation_matrix,
        n_iter=50000
    )
    
    print(f"\nComparison:")
    print(f"Initial ELN price (t=0): {results['eln_price']:.6f}")
    print(f"Current ELN price (t=1): {results_t1['eln_price_at_t']:.6f}")
    