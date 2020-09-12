# Simple zero-coupon bond price calculator

def zero_coupon_bond(par, y, t):
    return par/(1+y)**t

# print(zero_coupon_bond(100, 0.05, 5))
# prints: 78.35261664684589

# Writing the yield curve bootstrapping class

import math

class BootstrapYieldCurve(object):

    def __init__(self):
        self.zero_rates = dict()
        self.instruments = dict()

    def add_instrument(self, par, T, coup, price, compounding_freq=2):
        self.instruments[T] = (par, coup, price, compounding_freq)

    def get_maturities(self):
        return sorted(self.instruments.keys())

    def get_zero_rates(self):
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()
        return [self.zero_rates[T] for T in self.get_maturities()]

    def bootstrap_zero_coupons(self):
        for (T, instrument) in self.instruments.items():
            (par, coup, price, freq) = instrument
            if coup == 0:
                spot_rate = self.zero_coupon_spot_rate(par, price, T)
                self.zero_rates[T] = spot_rate

    def zero_coupon_spot_rate(self, par, price, T):
        spot_rate = math.log(par/price) / T
        return spot_rate

    def get_bond_spot_rates(self):
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par, coup, price, freq) = instrument
            if coup != 0:
                spot_rate = self.calculate_bond_spot_rate(T, instrument)
                self.zero_rates[T] = spot_rate

    def calculate_bond_spot_rate(self, T, instrument):
        try:
            (par, coup, price, freq) = instrument
            periods = T * freq
            value = price
            per_coupon = coup / freq
            for i in range(int(periods) - 1):
                t = (i+1)/float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * math.exp(-spot_rate*t)
                value -= discounted_coupon
            last_period = int(periods)/float(freq)
            spot_rate = -math.log(value/(par+per_coupon))/last_period
            return spot_rate
        except:
            print(f"Error: spot rate not found for T= {t}")

# Instantiate and add instruments:
yield_curve = BootstrapYieldCurve()
yield_curve.add_instrument(100, 0.25, 0., 97.5)
yield_curve.add_instrument(100, 0.5, 0., 94.9)
yield_curve.add_instrument(100, 1.0, 0., 90.)
yield_curve.add_instrument(100, 1.5, 8, 96., 2)
yield_curve.add_instrument(100, 2., 12, 101.6, 2)
y = yield_curve.get_zero_rates()
x = yield_curve.get_maturities()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
plt.plot(x,y)
plt.title("Zero Curve")
plt.ylabel('Zero Rate (%)')
plt.xlabel('Maturity in Years')


#%% Forward rates

# Generate a list of forward rates from a list of spot rates

class ForwardRates(object):

    def __init__(self):
        self.forward_rates = []
        self.spot_rates = dict()

    def add_spot_rates(self, T, spot_rate):
        self.spot_rates[T] = spot_rate

    def get_forward_rates(self):
        periods = sorted(self.spot_rates.keys())
        for T2, T1 in zip(periods, periods[1:]):
            forward_rate = self.calculate_forward_rate(T1, T2)
            self.forward_rates.append(forward_rate)

        return self.forward_rates

    def calculate_forward_rate(self, T1, T2):
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = (R2*T2 - R1*T1) / (T2-T1)
        return forward_rate

# Putting in spot rates:
fr = ForwardRates()
fr.add_spot_rates(0.25, 10.127)
fr.add_spot_rates(0.5, 10.469)
fr.add_spot_rates(1.00, 10.536)
fr.add_spot_rates(1.5, 10.681)
fr.add_spot_rates(2.00, 10.808)

print(fr.get_forward_rates())
# prints: [10.810999999999998, 10.603, 10.971, 11.189]

#%% Calculating Yield to maturity (YTM)
import scipy.optimize as optimize

def bond_ytm(price, par, T, coup, freq=2, guess=0.001):
    periods = T*freq
    coupon = coup / 100.*par
    dt = [(i+1)/freq for i in range(int(periods))]
    ytm_func = lambda y: \
        sum([(coupon/freq)/(1+y/freq)**(freq*t) for t in dt]) + \
        par / (1+y/freq)**(freq*T) - price

    return optimize.newton(ytm_func, guess)

ytm = bond_ytm(95.0428, 100, 1.5, 5.75, 2)
print(ytm)
# Prints: 0.09369155345237921

#%% Calculating the price of a bond

def bond_price(par, T, ytm, coup, freq=2):
    periods = T * freq
    coupon = coup / 100 * par
    dt = [(i+1)/freq for i in range(int(periods))]
    price = sum([coupon/freq/(1+ytm/freq)**(freq*t) for t in dt]) + \
        par/(1+ytm/freq)**(freq*T)
    return price

price = bond_price(100, 1.5, ytm, 5.75, 2)
print(price)
# Prints: 95.0428000000021

#%% Bond Duration

def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)
    # Get P -
    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    # Get P +
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    # Calculate modified duration
    mduration = (price_minus - price_plus) / (2*price*dy)
    return mduration

mod_duration = bond_mod_duration(95.0428, 100, 1.5, 5.75, 2)
print(mod_duration)
# prints: 1.3921935426561558

#%% Bond convexity

def bond_convexity(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)
    # Get P -
    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coup, freq)
    # Get P +
    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)
    # Calculate convexity
    convexity = (price_minus + price_plus - 2*price) / (price*dy**2)
    return convexity

convexity = bond_convexity(95.0428, 100, 1.5, 5.75, 2)
print(convexity)

#%% Short rate modelling

# The Vasicek model

import math
import numpy as np

def vasicek(r0, K, theta, sigma, T=1, N=10, seed=777):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + \
            sigma*math.sqrt(dt)*np.random.normal()
        rates.append(rates[-1]+dr)
    return range(N+1), rates

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
for K in [0.002, 0.02, 0.2, 2]:
    x, y = vasicek(0.005, K, 0.15, 0.05, T=10, N=200)
    ax.plot(x, y, label=f'K={K}')

ax.legend(loc='upper left')
ax.set_xlabel('Vasicek model');

#%% The Cox-Ingersoll-Ross model

import math
import numpy as np

def CIR(r0, K, theta, sigma, T=1, N=10, seed=777):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + \
            sigma*math.sqrt(rates[-1])* \
            math.sqrt(dt)*np.random.normal()
        rates.append(rates[-1]+dr)
    return range(N+1), rates

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
for K in [0.002, 0.02, 0.2, 2]:
    x, y = CIR(0.005, K, 0.15, 0.05, T=10, N=200)
    ax.plot(x, y, label=f'K={K}')

ax.legend(loc='upper left')
ax.set_xlabel('CIR model');

#%% The Brennan and Schwartz model

import math
import numpy as np

def brennan_schwartz(r0, K, theta, sigma, T=1, N=10, seed=1):
    np.random.seed(seed)
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K * (theta - rates[-1]) * dt + \
            sigma*rates[-1]* \
            math.sqrt(dt)*np.random.normal()

        rates.append(rates[-1] + dr)

    return range(N+1), rates

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
for K in [0.002, 0.02, 0.2]:
    x, y = brennan_schwartz(0.005, K, 0.006, 0.05, T=10, N=200)
    ax.plot(x, y, label=f'K={K}')

ax.legend(loc='upper left')
ax.set_xlabel('Brennan and Schwartz model');

#%% Pricing a zero-coupon bond by the Vasicek model

def exact_zcb(theta, kappa, sigma, tau, r0 = 0):
    B = (1 - np.exp(-kappa*tau)) / kappa
    A = np.exp((theta - (sigma**2) / (2*(kappa**2))) * (B-tau) - \
                (sigma**2) / (4*kappa)*(B**2))
    return A * np.exp(-r0*B)

Ts = np.r_[0:25.5:0.2]
zcbs = [exact_zcb(0.5, 0.02, 0.03, t, 0.015) for t in Ts]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12,8))
ax.set_title("Zero Coupon Bond (ZCB) Values by Time")
ax.plot(Ts, zcbs, label='ZCB')
ax.set_ylabel('Value ($)')
ax.set_xlabel('Time in years')
ax.legend()
ax.grid(True);
