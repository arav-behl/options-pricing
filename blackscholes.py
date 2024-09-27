from numpy import exp, sqrt, log, pi
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholes:
def __init__(
self,
time_to_maturity: float,
strike: float,
current_price: float,
volatility: float,
interest_rate: float,
dividend_yield: float = 0.0,
):
self.time_to_maturity = time_to_maturity
self.strike = strike
self.current_price = current_price
self.volatility = volatility
self.interest_rate = interest_rate
self.dividend_yield = dividend_yield
        self.implied_volatility = None

def run(self):
S = self.current_price
K = self.strike
T = self.time_to_maturity
r = self.interest_rate
q = self.dividend_yield
sigma = self.volatility

d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

self.call_price = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
self.put_price = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

# Greeks
self.call_delta = exp(-q * T) * norm.cdf(d1)
self.put_delta = -exp(-q * T) * norm.cdf(-d1)

self.call_gamma = exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))
self.put_gamma = self.call_gamma

self.call_vega = S * exp(-q * T) * norm.pdf(d1) * sqrt(T)
self.put_vega = self.call_vega

self.call_theta = -S * exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt(T)) - \
r * K * exp(-r * T) * norm.cdf(d2) + q * S * exp(-q * T) * norm.cdf(d1)
self.put_theta = -S * exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt(T)) + \
r * K * exp(-r * T) * norm.cdf(-d2) - q * S * exp(-q * T) * norm.cdf(-d1)

self.call_rho = K * T * exp(-r * T) * norm.cdf(d2)
self.put_rho = -K * T * exp(-r * T) * norm.cdf(-d2)

 # Implied Volatility
self.implied_volatility = self.calculate_implied_volatility()

def calculate_pnl(self, purchase_price, option_type='call'):
if option_type == 'call':
return self.call_price - purchase_price
elif option_type == 'put':
return self.put_price - purchase_price
else:
raise ValueError("option_type must be 'call' or 'put'")

def calculate_implied_volatility(self, market_price=None, option_type='call'):
if market_price is None:
market_price = self.call_price if option_type == 'call' else self.put_price

def option_price_diff(sigma):
            self.volatility = sigma
            self.run()
            model_price = self.call_price if option_type == 'call' else self.put_price
            temp_model = BlackScholes(
                self.time_to_maturity,
                self.strike,
                self.current_price,
                sigma,
                self.interest_rate,
                self.dividend_yield
            )
            temp_model.run()
            model_price = temp_model.call_price if option_type == 'call' else temp_model.put_price
return model_price - market_price

try:
implied_vol = brentq(option_price_diff, 1e-6, 10)
return implied_vol
except ValueError:
return None

if __name__ == "__main__":
# Test the BlackScholes class
bs = BlackScholes(
time_to_maturity=1,
strike=100,
current_price=100,
volatility=0.2,
interest_rate=0.05,
dividend_yield=0.02
)
bs.run()
print(f"Call Price: {bs.call_price:.4f}")
print(f"Put Price: {bs.put_price:.4f}")
print(f"Call Delta: {bs.call_delta:.4f}")
print(f"Put Delta: {bs.put_delta:.4f}")
print(f"Gamma: {bs.call_gamma:.4f}")
print(f"Vega: {bs.call_vega:.4f}")
print(f"Call Theta: {bs.call_theta:.4f}")
print(f"Put Theta: {bs.put_theta:.4f}")
print(f"Call Rho: {bs.call_rho:.4f}")
print(f"Put Rho: {bs.put_rho:.4f}")
    print(f"Implied Volatility: {bs.implied_volatility:.4f}")
    
    # Calculate implied volatility
    implied_vol = bs.calculate_implied_volatility()
    print(f"Implied Volatility: {implied_vol:.4f}")
