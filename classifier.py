import numpy as np

from solvers import GradientDescentSolver, StochasticGradientDescentSolver
from loss_functions import LossSVM2, LossSVM3


class SvmClassifier:
    """ A simple classifier that returns -1, or +1. """

    def __init__(self, lambd, eta):
        self.w = None
        self.ws = None
        self.losses = None
        self.lambd = lambd
        self.eta = eta

    def fit(self, xis, yis, N=1000):
        """ Fit using regular GDS. """
        lf = LossSVM2(lambd=self.lambd, xis=xis, yis=yis)
        gds = GradientDescentSolver(eta=self.eta, N=N, loss_function=lf)

        dim = xis.shape[1] + 1
        self.losses, self.ws = gds.solve(np.random.rand(dim))
        self.w = self.ws[-1]

    def fit_sgds(self, xis, yis, N=1000, batch_size=1):
        """ Fit using stochastic GDS. """
        lf = LossSVM3(lambd=self.lambd)
        sgds = StochasticGradientDescentSolver(loss_function=lf, xis=xis, yis=yis)

        dim = xis.shape[1] + 1
        start_w = np.random.rand(dim)
        self.losses, self.ws = sgds.solve(start_w, self.eta, N, batch_size)
        self.w = self.ws[-1]

    def decision_boundary(self, xi):
        w0 = self.w[0]
        w1 = self.w[1:]
        return w1.dot(xi) - w0

    def predict(self, xis):
        num_data_points = xis.shape[0]
        w0 = self.w[0]
        w1 = self.w[1:]
        yis = np.zeros(num_data_points)
        for i in range(num_data_points):
            xi = xis[i, :]
            if w1.dot(xi) - w0 > 0:
                yis[i] = 1
            else:
                yis[i] = -1
        return yis
