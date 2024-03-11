import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

class DRV:
    """ A model for discrete random variables where outcomes are numeric """
    def __init__(self, dist=None, type = 'discrete', min_val=0, max_val=10, mean=1, stddev=0, bins=20):
        self.type = type

        if self.type == 'discrete':
            if dist is None:
                self.dist = {}  # outcome -> p(outcome)
            else:
                self.dist = copy.deepcopy(dist)

        if self.type == 'uniform':
            self.bins = bins
            self.min_val = min_val
            self.max_val = max_val
            self.dist = {}
            x_ticks = np.linspace(self.min_val, self.max_val, bins)
            for x in x_ticks:
                self.dist[x] = 1/bins

        if self.type == 'normal':
            self.bins = bins
            self.mean = mean
            self.stddev = stddev
            self.dist = {}

            values = (np.random.normal(self.mean, self.stddev, self.bins *10)) #self.bins *10

            lower_bound = values.min()
            upper_bound = values.max()
            total = len(values)

            x_ticks = np.linspace(lower_bound, upper_bound, self.bins)

            for val in values:
                fit = min(x_ticks, key=lambda x: abs(x - val))
                if fit not in self.dist:
                    self.dist[fit] = 1
                else:
                    self.dist[fit] += 1

            for key in self.dist:
                self.dist[key] /= total

    def expected_value(self):
        eval = 0
        for key, value in self.dist.items():
            eval += (key * value)
        return eval

    def variance(self):
        mean = self.expected_value()

        var = 0
        for key, value in self.dist.items():
            var += ((key - mean) ** 2) * value
        return var

    def standard_deviation(self):
        var = self.variance()
        stddev = var ** .5
        return stddev

    def range(self):
        min_val = min(self.dist.keys())
        max_val = max(self.dist.keys())
        return max_val - min_val

    def random(self):
        """Sample from the discrete distribution."""
        outcomes = list(self.dist.keys())
        probabilities = list(self.dist.values())
        return random.choices(outcomes, weights=probabilities)[0]

    def __getitem__(self, x):
        return self.dist.get(x, 0.0)

    def __setitem__(self, x, p):
        self.dist[x] = p

    def apply(self, other, op):
        """ Apply a binary operator to self and other """
        Z = DRV()
        items = self.dist.items()
        oitems = other.dist.items()
        for x, px in items:
            for y, py in oitems:
                Z[op(x, y)] += px * py
        return Z

    def applyscalar(self, a, op):
        Z = DRV()
        items = self.dist.items()
        for x, p in items:
            Z[op(x,a)] += p
        return Z

    def __add__(self, other):
        return self.apply(other, lambda x, y: x + y)

    def __radd__(self, a):
        return self.applyscalar(a, lambda x, c: c + x)

    def __rmul__(self, a):
        return self.applyscalar(a, lambda x, c: c * x)

    def __rsub__(self, a):
        return self.applyscalar(a, lambda x, c: c - x)

    def __sub__(self, other):
        return self.apply(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.apply(other, lambda x, y: x * y)

    def __truediv__(self, other):
        # might require div by 0 handling
        return self.apply(other, lambda x, y: x / y)

    def __pow__(self, other):
        return self.apply(other, lambda x, y: x ** y)

    def __repr__(self):
        xp = sorted(self.dist.items())
        rslt = ''
        for x, p in xp:
            rslt += str(round(x)) + " : " + str(round(p, 8)) + "\n"
        return rslt

    def plot(self, title = 'Distribution Plot', show_cumulative = False, trials = 0, bins = 20, fig_size = (10,10),
             log_scale = False ):

        plt.figure(figsize = fig_size)

        if trials == 0:
           plt.bar(self.dist.keys(), self.dist.values())

        else:
            sample = [self.random() for i in range(trials)]
            sns.displot(sample, kind = 'hist', stat = 'probability', bins = bins)

        if show_cumulative:
            new_dist = {}
            items = sorted(self.dist.items())
            cumulative_prob = 0
            for i, (x, px) in enumerate(items):
                cumulative_prob += px
                new_dist[x] = cumulative_prob
                items[i] = (x, cumulative_prob)

            x_values = [item[0] for item in items]
            cumulative_probs = [item[1] for item in items]

            # Plot the CDF
            plt.plot(x_values, cumulative_probs, color='red', linestyle='-')

        if log_scale:
            plt.yscale('log')

        plt.xlabel('Value x')
        plt.ylabel('Probability p(x)')
        plt.title(title)
        plt.show()



def main():

    #U = DRV(type = 'normal', mean = 10, stddev = 5, bins = 10)
    #U.plot(show_cumulative= True)

    R = DRV(type='uniform', min_val=1.5, max_val=3)
    Fp = DRV(type='normal', mean=1, stddev=.05)
    Ne = DRV(type='uniform', min_val=1, max_val=5)

    '''Fraction of planets that can support life on which life actually develops. I would estimate all of these planets 
    develop life. Life to be as simple as single cell organisms, and if an environment can support the development of 
    life, life will happen.'''
    F1 = DRV(type='normal', mean=1.0, stddev=.05)

    '''Fraction of planets with life that develop intelligent life. I am not optimistic that most planets with life 
    will develop intelligent life. Earth, for example, has millions of species, but only one could be considered
    intelligent. I would estimate that 1/100 plantes with life will develop intelligent life.'''
    Fi = DRV(type='normal', mean=.01, stddev=.005)

    '''I believe that about half of intelligent planets will develop interstellar radio communication. 
    I think the odds are good that these civilizations can develop the necessary technology, but I figured
    its possible that 1/2 plantes may either not reach the intelligence level or develop technology and fail.'''
    Fc = DRV(type='normal', mean=.50, stddev=.05)

    '''I estimated that an advanced civilization would last 1500 years. With advanced civilizations and 
    increased technology, comes weapons of mass destruction and use of a planets natural resources. 
    Unfortunately, from what we've seen on Earth (nuclear wars, extreme climate change), I believe
    that an advanced civilization cannot last more than a few thousand years.'''
    L = DRV(type='normal', mean=1500, stddev=500)

    N = R * Fp * Ne * F1 * Fi * Fc * L

   # N.plot(trials=20)

    e = N.expected_value()
    s = N.standard_deviation()

    print(e, s)



if __name__ == '__main__':
    main()

