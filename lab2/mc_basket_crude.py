import numpy as np 

def arithmetic_basket_call(S0, K, r, sigma, t, T, N, iter, rho):
    tau = T - t
    rhomat = np.full([N, N], rho)
    np.fill_diagonal(rhomat, 1)
    sigma_matrix = np.diag([sigma] * N)*rhomat*np.diag([sigma] * N)*tau
    L = np.linalg.cholesky(sigma_matrix)
    #print(L)
    price = 0
    for i in range(iter):
        Z = np.random.randn(N)
        S = S0*np.exp(np.ones(N)*r-np.diag(sigma_matrix)/2*tau+sigma*Z*np.sqrt(tau))
        payoff = np.maximum(np.mean(S) - K, 0)
        price = price + np.exp(-r * tau)/iter * payoff
     
    return price