import numpy as np
from scipy.stats import norm

def mc_arithmetic_basket_control_custom_weights(S0, weights, K, r, sigma_array, t, T, correlation_matrix, n_iter, seed=None):
    tau = T - t
    N_assets = len(S0)

    S0 = np.array(S0, dtype=float)
    weights = np.array(weights, dtype=float)        # these are the arithmetic weights (e.g. c_i / S0_i)
    sigma_array = np.array(sigma_array, dtype=float)
    corr = np.array(correlation_matrix, dtype=float)

    # Recover c_i (geometric weights that sum to 1) if weights = c_i / S0
    c_weights = weights * S0
    c_weights = c_weights / c_weights.sum()   # guard against tiny numerical error

    # Precompute objects
    logS0 = np.log(S0)
    # full covariance without T factor
    Sigma = np.outer(sigma_array, sigma_array) * corr

    # geometric log-variance and mean (for log G)
    sigma_geom_sq = c_weights @ (Sigma @ c_weights) * tau    # scalar
    mu_geom = np.sum(c_weights * (logS0 + (r - 0.5 * sigma_array**2) * tau))

    # analytic UNdiscounted E[(G_T - K)+]
    s = np.sqrt(sigma_geom_sq)
    if s > 0:
        d1 = (mu_geom + sigma_geom_sq - np.log(K)) / s
        d2 = (mu_geom - np.log(K)) / s
        control_theoretical = np.exp(mu_geom + 0.5*sigma_geom_sq) * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        # degenerate: no variance
        control_theoretical = max(np.exp(mu_geom) - K, 0.0)

    # prepare Cholesky of correlation once
    Lcorr = np.linalg.cholesky(corr)

    if seed is not None:
        np.random.seed(seed)

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
    beta_opt = cov_xy / var_y if var_y > 0 else 0.0

    # control-variate (all undiscounted)
    controlled = payoffs_arith - beta_opt * (payoffs_geom - control_theoretical)

    # discount once at the end
    price = np.exp(-r * tau) * np.mean(controlled)
    stderr = np.exp(-r * tau) * np.std(controlled, ddof=1) / np.sqrt(n_iter)
    print("B0 =", np.sum(weights * S0))
    print("mean_arith_undiscounted =", np.mean(payoffs_arith))
    print("mean_geom_undiscounted =", np.mean(payoffs_geom))
    print("control_theoretical_undiscounted =", control_theoretical)
    print("beta_opt =", beta_opt)
    print("C0_call =", np.exp(-r*tau) * np.mean(controlled))
    print("ELN_price =", np.exp(-r*tau) * (1.0 + np.mean(controlled)))
    return price, stderr



# Example usage and testing
if __name__ == "__main__":
    print("Testing General Monte Carlo Basket Option Pricing")
    print("=" * 50)
    
def eln_pricing_with_participation_rate(NA, c_i, S0, r, T, sigma_array, correlation_matrix, n_iter=100000, target_multiple=1.1):
    """
    Price ELN and find participation rate ρT for fair value.
    
    ELN Payoff: Φ(T) = NA × (1 + ρT × (B(T) - 1))^+
    where B(T) = Σ(w_i × S_i(T)) and w_i = c_i/S_i(0)
    
    Parameters:
    ----------
    NA : float
        Notional amount
    c_i : array_like
        Original portfolio weights
    S0 : array_like
        Initial stock prices
    r : float
        Risk-free rate
    T : float
        Time to maturity
    sigma_array : array_like
        Individual asset volatilities
    correlation_matrix : array_like
        Asset correlation matrix
    iter : int
        Monte Carlo iterations
    target_multiple : float
        Target ELN price as multiple of NA (default 1.1)
        
    Returns:
    -------
    dict
        Results including basket call price, participation rate, etc.
    """
    
    # Convert to arrays
    c_i = np.array(c_i)
    S0 = np.array(S0)
    
    
    # Modified weights for basket
    weights = c_i / S0
    
    print(f"Initial basket value B(0): {np.sum(weights*S0)}")
    print(f"sum of c_i: {np.sum(c_i)}")
    
    print(f"ELN Pricing Analysis")
    print(f"=" * 40)
    print(f"Notional Amount (NA): {NA}")
    print(f"Target ELN Price: {target_multiple} × NA = {target_multiple * NA}")
    print(f"Time to maturity: {T} years")
    print(f"Risk-free rate: {r:.3%}")
    print(f"Discount factor: e^(-rT) = {np.exp(-r*T):.6f}")
    print(f"Bond component: e^(-rT) × NA = {np.exp(-r*T) * NA:.6f}")
    
    print(f"\nBasket Details:")
    print(f"Modified weights (w_i = c_i/S_i(0)): {weights}")
    print(f"Sum of weights: {np.sum(weights):.6f}")
    print(f"Initial basket value B(0): {np.sum(weights * S0):.6f}")
    
    # Price the arithmetic basket call with K=1
    print(f"\nPricing arithmetic basket call with K={1}...")
    basket_call_price, basket_call_error = mc_arithmetic_basket_control_custom_weights(
        S0=S0, weights=weights, K=1.0, r=r, 
        sigma_array=sigma_array, t=0, T=T, 
        correlation_matrix=correlation_matrix, n_iter=n_iter
    )
    
    print(f"Arithmetic basket call price: {basket_call_price:.6f} ± {basket_call_error:.6f}")
    
    # Calculate required participation rate
    bond_component = np.exp(-r * T)
    required_basket_component = target_multiple - bond_component
    
    if basket_call_price > 0:
        participation_rate = required_basket_component / basket_call_price
        print(f"\nParticipation Rate Calculation:")
        print(f"Required basket component: {target_multiple} - {bond_component:.6f} = {required_basket_component:.6f}")
        print(f"Participation rate ρT: {required_basket_component:.6f} / {basket_call_price:.6f} = {participation_rate:.6f}")
        
        # Verify the ELN price
        eln_price = bond_component * NA + NA * participation_rate * basket_call_price
        print(f"\nVerification:")
        print(f"ELN Price = {bond_component:.6f} × {NA} + {NA} × {participation_rate:.6f} × {basket_call_price:.6f}")
        print(f"ELN Price = {eln_price:.6f}")
        print(f"Target was: {target_multiple * NA:.6f}")
        print(f"Match: {abs(eln_price - target_multiple * NA) < 1e-10}")
        
    else:
        participation_rate = None
        eln_price = None
        print(f"Warning: Basket call price is {basket_call_price}, cannot calculate participation rate")
    
    return {
        'basket_call_price': basket_call_price,
        'basket_call_error': basket_call_error,
        'participation_rate': participation_rate,
        'bond_component': bond_component * NA,
        'basket_component': NA * participation_rate * basket_call_price if participation_rate else None,
        'eln_price': eln_price,
        'target_price': target_multiple * NA,
        'weights': weights,
        'initial_basket_value': np.sum(weights * S0)
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing General Monte Carlo Basket Option Pricing")
    print("=" * 50)
    
    # ELN Test case with your data
    print("\nELN Pricing Test:")
    c_i = np.array([0.07, 0.11, 0.11, 0.18, 0.06, 0.04, 0.01, 0.17, 0.04, 0.02, 0.03, 0.16])
    S0 = np.array([141.3, 80.9, 193.3, 316.7, 212.9, 75.6, 236.6, 108.8, 20.68, 192.9, 46.55, 88.95])
    
    # Example volatilities and correlation matrix (you should use your actual data)
    sigma_array =  np.array([0.32, 0.22, 0.31, 0.18, 0.27, 0.22, 0.21, 0.23, 0.34, 0.17, 0.18, 0.23])
    
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


def eln_pricing_at_time_t(NA, participation_rate, c_i, S0_original, S_current, r, t, T, sigma_array, correlation_matrix, n_iter=100000):
    """
    Calculate ELN fair value at time t > 0, given fixed participation rate.
    
    The ELN payoff is still: Φ(T) = NA × (1 + ρT × (B(T) - 1))^+
    But now we need to price it from time t with remaining time τ = T - t
    
    Parameters:
    ----------
    NA : float
        Notional amount
    participation_rate : float
        Fixed participation rate ρT (from initial pricing)
    c_i : array_like
        Original portfolio weights (unchanged)
    S0_original : array_like
        Original stock prices at t=0 (for calculating weights)
    S_current : array_like
        Current stock prices at time t
    r : float
        Risk-free rate (assumed constant)
    t : float
        Current time
    T : float
        Original maturity time
    sigma_array : array_like
        Individual asset volatilities
    correlation_matrix : array_like
        Asset correlation matrix
    iter : int
        Monte Carlo iterations
        
    Returns:
    -------
    dict
        ELN pricing results at time t
    """
    
    tau = T - t  # Remaining time to maturity
    
    if tau <= 0:
        print("Warning: ELN has already matured!")
        return None
    
    # Convert to arrays
    c_i = np.array(c_i)
    S0_original = np.array(S0_original)
    S_current = np.array(S_current)
    
    # Modified weights are based on ORIGINAL prices (fixed at inception)
    weights = c_i / S0_original
    
    print(f"ELN Pricing at Time t = {t}")
    print(f"=" * 40)
    print(f"Notional Amount (NA): {NA}")
    print(f"Participation Rate (ρT): {participation_rate:.6f} (fixed from inception)")
    print(f"Current time: t = {t}")
    print(f"Original maturity: T = {T}")
    print(f"Remaining time: τ = {tau}")
    print(f"Risk-free rate: {r:.3%}")
    print(f"")
    
    print(f"Stock Price Changes:")
    print(f"Original prices (t=0): {S0_original}")
    print(f"Current prices (t={t}):  {S_current}")
    price_changes = (S_current / S0_original - 1) * 100
    print(f"Price changes (%):      {price_changes}")
    print(f"")
    
    print(f"Basket Details:")
    print(f"Modified weights (fixed): {weights}")
    print(f"Current basket value: {np.sum(weights * S_current):.6f}")
    
    # Simple approach: Price the ELN payoff from current state to maturity
    # We simulate B(T)/B(0) starting from current basket level
    # The payoff is max(B(T)/B(0) - 1, 0)
    
    current_basket_value = np.sum(weights * S_current)
    
    print(f"\\nPricing remaining basket call option...")
    print(f"Using current prices as 'initial' for remaining {tau}-year period")
    print(f"Current basket value: {current_basket_value:.6f}")
    
    # Price basket call with strike K=1 (since we want max(B(T)/B(0) - 1, 0))
    # Starting from current prices, simulating to maturity
    K_eff = np.sum(weights * S0_original) / np.sum(weights * S_current)
    basket_call_price, basket_call_error = mc_arithmetic_basket_control_custom_weights(
        S0=S_current, weights=weights, K=1.0, r=r,
        sigma_array=sigma_array, t=0, T=tau,  # Note: using tau as time to maturity
        correlation_matrix=correlation_matrix, n_iter=n_iter
    )
    
    print(f"Remaining basket call price: {basket_call_price:.6f} ± {basket_call_error:.6f}")
    
    # Calculate ELN components
    bond_component = np.exp(-r * tau) * NA
    basket_component = NA * participation_rate * basket_call_price
    eln_price_current = bond_component + basket_component
    
    print(f"\\nELN Valuation at t = {t}:")
    print(f"Bond component: e^(-r×{tau}) × {NA} = {bond_component:.6f}")
    print(f"Basket component: {NA} × {participation_rate:.6f} × {basket_call_price:.6f} = {basket_component:.6f}")
    print(f"Total ELN Price: {eln_price_current:.6f}")
    print(f"As multiple of NA: {eln_price_current/NA:.6f} × NA")
    
    return {
        'eln_price_at_t': eln_price_current,
        'bond_component': bond_component,
        'basket_component': basket_component,
        'remaining_basket_call_price': basket_call_price,
        'remaining_basket_call_error': basket_call_error,
        'current_basket_value': current_basket_value,
        'remaining_time': tau,
        'price_changes_percent': price_changes,
        'current_time': t
    }


# Example usage for t=1 pricing
if __name__ == "__main__":
    print("Testing General Monte Carlo Basket Option Pricing")
    print("=" * 50)
    
    # ELN Test case with your data (using same parameters as first test)
    print("\\nELN Pricing Test:")
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
    
    print(f"\\n" + "="*60)
    print("ELN PRICING AFTER 1 YEAR (t=1)")
    print("="*60)
    
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
    
    print(f"\\nComparison:")
    print(f"Initial ELN price (t=0): {results['eln_price']:.6f}")
    print(f"Current ELN price (t=1): {results_t1['eln_price_at_t']:.6f}")
    print(f"Change in value: {results_t1['eln_price_at_t'] - results['eln_price']:.6f}")
    print(f"Percentage change: {((results_t1['eln_price_at_t'] / results['eln_price']) - 1)*100:.2f}%")
    print(f"S_current: {S_current}")