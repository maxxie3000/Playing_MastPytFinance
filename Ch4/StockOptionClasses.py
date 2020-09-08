import math

class StockOption(object):
    def __init__(self, S0, K, r=0.05, T=1, N=2, pu=0, pd=0,
                 div=0, sigma=0, is_put=False, is_am=False):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1, N)
        self.STs = [] # Declare the stock prices tree

        # Optional parameters used by derived classes
        self.pu, self.pd = pu, pd
        self.div = div
        self.sigma = sigma
        self.is_call = not is_put
        self.is_european = not is_am

    @property
    def dt(self):
            '''Single time step, in years'''
            return self.T/float(self.N)

    @property
    def df(self):
            '''The discount factor'''
            return math.exp(-(self.r-self.div)*self.dt)

# A class for Europen options using a binomial tree
import numpy as np
from decimal import Decimal

class BinomialEuropeanOption(StockOption):
    def setup_parameters(self):
        # Required calculations for the model
        self.M = self.N+1 # Number of terminal nodes of tree
        self.u = 1+self.pu # Expected value in the up state
        self.d = 1-self.pd # Expected value in the down state
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def init_stock_price_tree(self):
        # Initialize terminal price nodes to zeros
        self.STs = np.zeros(self.M)
        # Calculate expected stock prices for each node
        for i in range(self.M):
            self.STs[i] = self.S0 * (self.u**(self.N-i)) * (self.d**i)

    def init_payoffs_tree(self):
        """
        Returns the payoffs when the option expires at terminal nodes
        """
        if self.is_call:
            return np.maximum(0, (self.STs - self.K))
        else:
            return np.maximum(0, (self.K - self.STs))

    def traverse_tree(self, payoffs):
        """
        Starting from the time the option expires, traverse backwards and
        calculate discounted payoffs at each node
        """
        for i in range(self.N):
            payoffs = (payoffs[:-1]*self.qu +
                       payoffs[1:]*self.qd) * self.df

        return payoffs

    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        """ Entry point of the pricing implementation """
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        # Option value converges to first node
        return payoffs[0]

# eu_opt = BinomialEuropeanOption(50, 52, r=0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True)

# print(f'EU put option price is: {eu_opt.price()}')

# Class for pricing American and European options
class BinomialTreeOption(StockOption):
    def setup_parameters(self):
        # Required calculations for the model
        self.u = 1+self.pu # Expected value in the up state
        self.d = 1-self.pd # Expected value in the down state
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def init_stock_price_tree(self):
        # Initialize a 2D tree at T=0
        self.STs = [np.array([self.S0])]
        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                ( prev_branches * self.u,
                [prev_branches[-1] * self.d])
                )
            self.STs.append(st) # Add nodes at each time step
            # print(self.STs)

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, (self.STs[self.N] - self.K))
        else:
            return np.maximum(0, (self.K - self.STs[self.N]))

    def check_early_exercise(self, payoffs, node):
        """
        Returns the maximum payoff values between exercising the American options
        early and not exercising the option at all
        """
        if self.is_call:
            return np.maximum(payoffs, self.STs[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])

    def traverse_tree(self, payoffs):
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-1]*self.qu +
                       payoffs[1:]*self.qd) * self.df

            # Added check early exercise
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs, i)

        return payoffs

    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        """ Entry point of the pricing implementation """
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        # Option value converges to first node
        return payoffs[0]

#am_opt = BinomialTreeOption(50, 52, r=0.05, T=4, N=2, pu=0.2, pd=0.2, is_put=True, is_am=True)

#print(f'American put option price is: {am_opt.price()}')
#Option pricing by the binomial CRR model
class binomialCRROption(BinomialTreeOption):
    def setup_parameters(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r - self.div) * self.dt) -
                    self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

# eu_opt = binomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)

# am_opt = binomialCRROption(50, 52, r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)

# print(f'European option: {eu_opt.price()}, American option: {am_opt.price()}')

# Output: European option: 6.245708445206436, American option: 7.428401902704834

# Class for Leisen-Reimer tree
class BinomialLROption(BinomialTreeOption):

    def setup_parameters(self):
        odd_N = self.N if (self.N%2 == 0) else (self.N+1)

        d1 = (math.log(self.S0/self.K) + ((self.r-self.div) +
                (self.sigma**2)/2.) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = (math.log(self.S0/self.K) + ((self.r-self.div) -
                (self.sigma**2)/2. ) * self.T) / (self.sigma * math.sqrt(self.T))

        pbar = self.pp_2_inversion(d1, odd_N)
        self.p = self.pp_2_inversion(d2, odd_N)
        self.u = 1 / self.df * pbar / self.p
        self.d = (1 / self.df - self.p * self.u) / (1 - self.p)
        self.qu = self.p
        self.qd = 1-self.p

    def pp_2_inversion(self, z, n):
        return .5 + math.copysign(1, z) * \
            math.sqrt(.25 - .25 *
                math.exp(
                    -((z / (n + 1./3. + .1/(n+1)))**2.) * (n + 1./6.)
                )
            )

# eu_option = BinomialLROption(50, 52, r=0.05, T=2, N=4, sigma=0.3, is_put=True)
#
# am_option = BinomialLROption(50, 52, r=0.05, T=2, N=4, sigma=0.3, is_put=True, is_am=True)
#
# print(f'EU put option: {eu_option.price()}, AM put option: {am_option.price()}')

# Output: EU put option: 5.878650106601964, AM put option: 6.763641952939979

# Compute option price, delta and gamma by the LR begin_tree_traversal
class BinomialLRWithGreeks(BinomialLROption):
    def new_stock_price_tree(self):
        # Creates an additional layer of nodes to our original stock price begin_tree_traversal
        self.STs = [np.array([self.S0*self.u/self.d,
                              self.S0,
                              self.S0*self.d/self.u])]

        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches*self.u,
                                 [prev_branches[-1]*self.d]))

            self.STs.append(st)

    def price(self):
        self.setup_parameters()
        self.new_stock_price_tree()
        payoffs = self.begin_tree_traversal()

        # Option value is now in the middle node at t=0
        option_value = payoffs[len(payoffs)//2]

        payoff_up = payoffs[0]
        payoff_down = payoffs[-1]

        S_up = self.STs[0][0]
        S_down = self.STs[0][-1]

        dS_up = S_up - self.S0
        dS_down = self.S0 - S_down

        # Calculate delta value
        dS = S_up - S_down
        dV = payoff_up - payoff_down
        delta = dV/dS

        # Calculate gamma value
        gamma = ((payoff_up - option_value)/dS_up -
                 (option_value - payoff_down)/dS_down) / \
                 ((self.S0 + S_up)/2. - (self.S0 + S_down) / 2.)

        return option_value, delta, gamma

# eu_call = BinomialLRWithGreeks(50, 52, r=0.05, T=2, N=300, sigma=0.3)
# results = eu_call.price()
# print(f'European call values \nPrice: {results[0]} \nDelta: {results[1]} \nGamma: {results[2]}')
#
# am_option = BinomialLRWithGreeks(50, 52, r=0.05, T=2, N=300, sigma=0.3, is_put=True)
# results = am_option.price()
# print(f'American call values \nPrice: {results[0]} \nDelta: {results[1]} \nGamma: {results[2]}')

# prints:
# European call values
# Price: 9.69546807138366
# Delta: 0.6392477816643529
# Gamma: 0.01764795890533088
# American call values
# Price: 6.747013809252746
# Delta: -0.3607522183356649

# Class for trinomial tree option pricing model

class TrinomialTreeOption(BinomialTreeOption):

    def setup_parameters(self):
        # Required calculations for the model
        self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
        self.d = 1/self.u
        self.m = 1

        self.qu = ((math.exp((self.r-self.div) *
                             self.dt/2.) -
                    math.exp(-self.sigma *
                             math.sqrt(self.dt/2.))) /
                   (math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) -
                    math.exp(-self.sigma *
                             math.sqrt(self.dt/2.))))**2.

        self.qd = ((math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) -
                    math.exp((self.r-self.div) *
                             self.dt/2.)) /
                   (math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) -
                    math.exp(-self.sigma *
                             math.sqrt(self.dt/2.))))**2.

        self.qm = self.m - self.qu - self.qd

    def init_stock_price_tree(self):
        # Initialize a 2D tree at t=0
        self.STs = [np.array([self.S0])]

        for i in range(self.N):
            prev_nodes = self.STs[-1]
            self.ST = np.concatenate((prev_nodes*self.u, [prev_nodes[-1]*self.m,
                                                          prev_nodes[-1]*self.d]))
            self.STs.append(self.ST)

    def traverse_tree(self, payoffs):
        # Traverse the tree backwards
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-2] * self.qu +
                       payoffs[1:-1] * self.qm +
                       payoffs[2:] * self.qd) * self.df

            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs, i)

        return payoffs

# eu_put = TrinomialTreeOption(50, 52, T=2, sigma=0.3, is_put=True)
# am_put = TrinomialTreeOption(50, 52, T=2, sigma=0.3, is_put=True, is_am=True)
# print(f'EU put price: {eu_put.price()}, AM put price: {am_put.price()}')
# prints: EU put price: 6.573565269142496, AM put price: 7.161349217272585

# Class for CRR binomial lattice option pricing

class BinomialCRRLattice(binomialCRROption):

    def setup_parameters(self):
        super(BinomialCRRLattice, self).setup_parameters()
        self.M = 2*self.N + 1

    def init_stock_price_tree(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0 * self.u**self.N

        for i in range(self.M)[1:]:
            self.STs[i] = self.STs[i-1]*self.d

    def init_payoffs_tree(self):
        odd_nodes = self.STs[::2] # Take odd nodes only
        if self.is_call:
            return np.maximum(0, odd_nodes-self.K)
        else:
            return np.maximum(0, self.K-odd_nodes)

    def check_early_exercise(self, payoffs, node):
        self.STs = self.STs[1:-1]
        odd_STs = self.STs[::2]
        if self.is_call:
            return np.maximum(payoffs, odd_STs - self.K)
        else:
            return np.maximum(payoffs, self.K - odd_STs)

# eu_opt = BinomialCRRLattice(50,52, r=0.05, T=2, N=2, sigma=0.3, is_put=True)
# am_opt = BinomialCRRLattice(50,52, r=0.05, T=2, N=2, sigma=0.3, is_put=True, is_am=True)
# print(f'EU put price: {eu_opt.price()}, AM put price: {am_opt.price()}')
# prints: EU put price: 6.245708445206432, AM put price: 7.428401902704828
