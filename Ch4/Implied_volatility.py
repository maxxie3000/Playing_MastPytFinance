from StockOptionClasses import BinomialLROption
import bisect

# Get implied volatilities from a Leisen-Reimer binomial tree using the
# bisection method as the numerical procedure

class ImpliedVolatilityModel(object):

    def __init__(self, S0, r=0.05, T=1, div=0, N=1, is_put=False):
        self.S0 = S0
        self.r = r
        self.T = T
        self.div = div
        self.N = N
        self.is_put = is_put

    def option_valuation(self, K, sigma):
        # Use the binomial Leisen-Reimer tree
        lr_option = BinomialLROption(
            self.S0, K, r=self.r, T=self.T, N=self.N, sigma=sigma,
            div=self.div, is_put=self.is_put
        )
        return lr_option.price()

    def get_implied_volatilities(self, Ks, opt_prices):
        impvols = []
        for i in range(len(strikes)):
            # Bind f(sigma) for use by the bisection method
            f = lambda sigma: self.option_valuation(Ks[i], sigma) - opt_prices[i]
            impv = self.bisection(f, 0.01, 0.99, 0.0001, 100)[0]
            impvols.append(impv)
        return impvols

    def bisection(self, f, a, b, tol=0.1, maxiter=10):
        c = (a+b)*0.5 # C as the midpoint of a and b
        n = 1
        while n <= maxiter:
            c = (a+b)*0.5
            if f(c) == 0 or abs(a-b)*0.5 < tol:
                # Root is found or is very close
                return c, n

            n += 1
            if f(c) < 0:
                a = c
            else:
                b = c
        return c, n


strikes = [75, 80, 85, 90, 92.5, 95, 97.5,
           100, 105, 110, 115, 120, 125]
put_prices = [0.16, 0.32, 0.6, 1.22, 1.77, 2.54, 3.55,
              4.8, 7.75, 11.8, 15.96, 20.75, 25.81]


model = ImpliedVolatilityModel(
    99.62, r=0.0248, T=78/365., div=0.0182, N=77, is_put=True)
impvols_put = model.get_implied_volatilities(strikes, put_prices)

import matplotlib.pyplot as plt

plt.plot(strikes, impvols_put)
plt.xlabel('Strike Prices')
plt.ylabel('Implied Volatilities')
plt.title('AAPL Put Implied Volatilities expiring in 78 days')
plt.show()
