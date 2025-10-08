"""
Proof of Arbitrage Bounds for Arithmetic Basket Call Options

We prove: P^c_GB(0,T,K) ≤ P^c_AB(0,T,K) ≤ ∑c_i P^c_Ei(0,T,K)

where:
- P^c_GB = Geometric basket call price
- P^c_AB = Arithmetic basket call price  
- P^c_Ei = European call price on asset i
- c_i are positive weights summing to 1
- ∑c_i K_i = K (strike constraint)
"""

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes call option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def geometric_basket_call_analytical(S0, weights, K, T, r, sigma_matrix):
    """
    Analytical price of geometric basket call option
    """
    n = len(S0)
    weights = np.array(weights)
    
    # Calculate effective parameters for geometric basket
    log_S_geom = np.sum(weights * np.log(S0))
    S_geom = np.exp(log_S_geom)
    
    # Effective variance: w^T Σ w where Σ is covariance matrix of log returns
    sigma_geom_squared = weights.T @ sigma_matrix @ weights * T
    sigma_geom = np.sqrt(sigma_geom_squared)
    
    # Drift adjustment
    r_adj = r - 0.5*sigma_geom_squared + 0.5*np.sum(weights * np.diag(sigma_matrix))
    
    # Black-Scholes formula
    return black_scholes_call(S_geom, K, T, r_adj, sigma_geom)

def prove_lower_bound():
    """
    PROOF OF LOWER BOUND: P^c_GB(0,T,K) ≤ P^c_AB(0,T,K)
    
    This follows from Jensen's inequality and the convexity of max(x-K, 0).
    """
    print("="*60)
    print("PROOF OF LOWER BOUND")
    print("="*60)
    
    print("""
    THEOREM: P^c_GB(0,T,K) ≤ P^c_AB(0,T,K)
    
    PROOF:
    1) The payoff functions are:
       - Geometric: max(∏S_i(T)^c_i - K, 0) = max(exp(∑c_i log S_i(T)) - K, 0)
       - Arithmetic: max(∑c_i S_i(T) - K, 0)
    
    2) By Jensen's inequality, since exp is convex:
       exp(∑c_i log S_i(T)) ≤ ∑c_i S_i(T)
       
       This is because: ∏S_i^c_i ≤ ∑c_i S_i (weighted geometric ≤ weighted arithmetic mean)
    
    3) Since max(x-K, 0) is increasing in x:
       max(∏S_i^c_i - K, 0) ≤ max(∑c_i S_i - K, 0)
       
    4) Taking expectations and discounting:
       E[e^(-rT) max(∏S_i^c_i - K, 0)] ≤ E[e^(-rT) max(∑c_i S_i - K, 0)]
       
    5) Therefore: P^c_GB(0,T,K) ≤ P^c_AB(0,T,K) □
    """)

def prove_upper_bound():
    """
    PROOF OF UPPER BOUND: P^c_AB(0,T,K) ≤ ∑c_i P^c_Ei(0,T,K)
    
    This uses convexity of max(x-K, 0) and the constraint ∑c_i K_i = K.
    """
    print("="*60)
    print("PROOF OF UPPER BOUND")
    print("="*60)
    
    print("""
    THEOREM: P^c_AB(0,T,K) ≤ ∑c_i P^c_Ei(0,T,K_i) where ∑c_i K_i = K
    
    PROOF:
    1) The arithmetic basket payoff is:
       max(∑c_i S_i(T) - K, 0) = max(∑c_i S_i(T) - ∑c_i K_i, 0)
    
    2) Since max(x-K, 0) is convex in x, by Jensen's inequality:
       max(∑c_i S_i(T) - ∑c_i K_i, 0) ≤ ∑c_i max(S_i(T) - K_i, 0)
       
       This follows because:
       f(∑c_i x_i) ≤ ∑c_i f(x_i) when f is convex and ∑c_i = 1
    
    3) Taking expectations and discounting:
       E[e^(-rT) max(∑c_i S_i(T) - K, 0)] ≤ E[e^(-rT) ∑c_i max(S_i(T) - K_i, 0)]
       
    4) By linearity of expectation:
       P^c_AB(0,T,K) ≤ ∑c_i E[e^(-rT) max(S_i(T) - K_i, 0)]
       
    5) Therefore: P^c_AB(0,T,K) ≤ ∑c_i P^c_Ei(0,T,K_i) □
    """)

def numerical_verification():
    """
    Numerical verification of the bounds
    """
    print("="*60)
    print("NUMERICAL VERIFICATION")
    print("="*60)
    
    # Parameters
    S0 = np.array([100, 100, 100])  # Initial stock prices
    weights = np.array([0.4, 0.3, 0.3])  # Weights sum to 1
    K = 100  # Strike for arithmetic basket
    T = 1.0
    r = 0.05
    sigma = 0.2  # Individual volatilities
    rho = 0.3   # Correlation
    
    # Construct covariance matrix
    n = len(S0)
    sigma_matrix = np.full((n, n), rho * sigma**2 * T)
    np.fill_diagonal(sigma_matrix, sigma**2 * T)
    
    # For upper bound, we need K_i such that ∑c_i K_i = K
    # Simple choice: K_i = K for all i (then ∑c_i K_i = K ∑c_i = K)
    K_individual = np.full(n, K)
    
    # Calculate prices
    P_GB = geometric_basket_call_analytical(S0, weights, K, T, r, sigma_matrix)
    
    # For arithmetic basket, we'll use a simple approximation
    # (In practice, you'd use Monte Carlo)
    S_arith_approx = np.sum(weights * S0)
    sigma_arith_approx = sigma * np.sqrt(np.sum(weights**2) + 2*rho*np.sum([weights[i]*weights[j] 
                                                                         for i in range(n) for j in range(i+1,n)]))
    P_AB_approx = black_scholes_call(S_arith_approx, K, T, r, sigma_arith_approx)
    
    # Upper bound: sum of individual call options
    P_upper = sum(weights[i] * black_scholes_call(S0[i], K_individual[i], T, r, sigma) 
                  for i in range(n))
    
    print(f"Geometric Basket Call Price: {P_GB:.4f}")
    print(f"Arithmetic Basket Call Price (approx): {P_AB_approx:.4f}")
    print(f"Upper Bound (∑c_i P^c_Ei): {P_upper:.4f}")
    print(f"\nVerification:")
    print(f"P_GB ≤ P_AB: {P_GB:.4f} ≤ {P_AB_approx:.4f} → {P_GB <= P_AB_approx}")
    print(f"P_AB ≤ Upper: {P_AB_approx:.4f} ≤ {P_upper:.4f} → {P_AB_approx <= P_upper}")

def key_insights():
    """
    Key mathematical insights
    """
    print("="*60)
    print("KEY MATHEMATICAL INSIGHTS")
    print("="*60)
    
    print("""
    1. CONVEXITY IS CRUCIAL:
       - max(x-K, 0) is convex in x
       - Jensen's inequality: f(E[X]) ≤ E[f(X)] for convex f
       - This gives us both bounds!
    
    2. MONOTONICITY:
       - max(x-K, 0) is increasing in x
       - Geometric mean ≤ Arithmetic mean → lower bound
    
    3. ARBITRAGE INTERPRETATION:
       - Lower bound: Can't get geometric basket cheaper than arithmetic
       - Upper bound: Portfolio of individual calls dominates basket call
    
    4. ECONOMIC INTUITION:
       - Diversification reduces risk
       - Individual options have more optionality than basket
       - Correlation reduces the benefit of diversification
    
    5. PRACTICAL IMPLICATIONS:
       - Bounds are tight when assets are highly correlated
       - Monte Carlo needed for exact arithmetic basket price
       - Geometric basket has analytical solution (useful as control variate)
    """)

if __name__ == "__main__":
    prove_lower_bound()
    prove_upper_bound()
    numerical_verification()
    key_insights()