import numpy as np
import scipy.integrate as integrate
from scipy.optimize import brentq
from scipy.stats import norm
import warnings

# Suppress scipy integration warnings
warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

class HestonPricer:
    def __init__(self, kappa, theta, sigma, rho, v0, r=0.0, q=0.0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.q = q

    def characteristic_function(self, u, T):
        alpha = -u**2 / 2 - 1j * u / 2
        beta = self.kappa - self.rho * self.sigma * 1j * u
        gamma = self.sigma**2 / 2
        
        d = np.sqrt(beta**2 - 4 * alpha * gamma)
        r_plus = (beta + d) / (2 * gamma)
        r_minus = (beta - d) / (2 * gamma)
        g = r_minus / r_plus
        
        C = self.kappa * (r_minus * T - (2 / self.sigma**2) * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = r_minus * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        
        return np.exp(C * self.theta + D * self.v0)

    def price_european_call(self, S0, K, T):
        F = S0 * np.exp((self.r - self.q) * T)
        k = np.log(S0 / K) + (self.r - self.q) * T
        def integrand(u):
            cf = self.characteristic_function(u - 0.5j, T)
            return (np.exp(-1j * u * k) * cf / (u**2 + 0.25)).real
            
        integral, _ = integrate.quad(integrand, 0, np.inf, limit=1000)
        call_price = S0 * np.exp(-self.q * T) - K * np.exp(-self.r * T) * (1 / np.pi) * integral
        return max(call_price, 0.0)

def bs_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def implied_volatility(target_call_price, S0, K, T, r):
    """Robust Implied Volatility Solver using Put-Call Parity for ITM options"""
    # Convert Deep ITM Call to OTM Put for numerical stability
    if K < S0:
        target_price = target_call_price - S0 + K * np.exp(-r * T)
        is_call = False
    else:
        target_price = target_call_price
        is_call = True
        
    # If the OTM option price is practically zero, return a very small volatility to prevent NaN holes
    if target_price <= 1e-6:
        return 1e-4 
        
    def objective_function(sigma):
        if is_call:
            return bs_call_price(S0, K, T, r, sigma) - target_price
        else:
            return bs_put_price(S0, K, T, r, sigma) - target_price
            
    try:
        # Expanded bounds to handle extreme market parameters
        return brentq(objective_function, 1e-4, 10.0)
    except ValueError:
        return np.nan