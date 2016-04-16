import numpy as np


class OptimizableFunction:
    """ This is an interface for a function that may be optimized.
    Instead of using a numerical derivative, this interface expects one
    to know and write out an explicit expression for the derivative.
    """

    def eval_value(self, w):
        """ Result should be a scalar. """
        raise NotImplementedError("Not implemented yet")

    def eval_grad(self, w):
        """ Result should be a vector of dimensionality of input w. """
        raise NotImplementedError("Not implemented yet")


class LossSVM1(OptimizableFunction):
    """ First version of the loss function for an SVM.  This is not the full
    loss function, but just a starting point.

    In particular, in this version the hyperplane must pass through origin, i.e.
    there is no w_0 constant term. """
    def __init__(self, lambd, xis, yis):
        self.lambd = lambd
        self.xis = xis
        self.yis = yis

    def _check(self):
        assert self.xis.shape[0] == self.yis.shape[0]

    def eval_value(self, w):
        n = self.xis.shape[0]
        xis = self.xis
        yis = self.yis
        s = 0.0
        for i in range(n):
            s += max(0, 1 - yis[i] * xis[i, :].dot(w))
        return s / n + self.lambd*w.dot(w)

    def eval_grad(self, w):
        """ Result should be a vector of dimensionality of input w. """
        dim = w.shape[0]
        n = self.xis.shape[0]
        xis = self.xis
        yis = self.yis
        s = np.zeros(dim)
        for i in range(n):
            # NOTE: This is an explicit derivative of a max() function, and
            # as such must be split into 2 regions.
            # It is not exactly differentiable at point for which
            # yis[i]*xis[i,:].dot(w) = 1 holds true, and we choose to return
            # 0 there.  (It doesn't seem to be a problem for the numerical
            # stability. Hence the < below sign.)
            if yis[i]*xis[i, :].dot(w) < 1:
                s += -yis[i] * xis[i, :]
            # else: s += 0, but that is not necessary.
        return s / n + 2*self.lambd*w


class LossSVM2(OptimizableFunction):
    """ Second version of the loss function for an SVM (no kernel trick).

    In particular, in this version the hyperplane does not have to pass through
    the origin, i.e. there is a w_0 constant term. """
    def __init__(self, lambd, xis, yis):
        self.lambd = lambd
        self.xis = xis
        self.yis = yis

    def _check(self):
        assert self.xis.shape[0] == self.yis.shape[0]

    def eval_value(self, w):
        """ Result is a scalar. """
        self._check()

        w0 = w[0]
        w1 = w[1:]

        n = self.xis.shape[0]
        xis = self.xis
        yis = self.yis
        s = 0.0
        for i in range(n):
            s += max(0, 1 - yis[i] * (xis[i, :].dot(w1) - w0))
        return s / n + self.lambd*w1.dot(w1)

    def eval_grad(self, w):
        """ Result should be a vector of dimensionality of input w. """

        self._check()

        w0 = w[0]
        w1 = w[1:]

        dim = w.shape[0]
        n = self.xis.shape[0]
        xis = self.xis
        yis = self.yis
        s = np.zeros(dim)
        for i in range(n):
            # Note, like above, we handle the derivative of max() below, but we
            # must also handle w[0]' case specially.
            if yis[i]*(xis[i, :].dot(w1) - w0) < 1:
                s[0] += yis[i]*w0
                s[1:] += -yis[i] * xis[i, :]
            # else: s += 0, but that is not necessary.
        s /= n
        s[1:] += 2*self.lambd*w1
        return s


class LossSVM3(OptimizableFunction):
    """ Change the interface of LossSVM to pass in xis and yis as unfortunately
    the interface doesn't fit the solver. """
    def __init__(self, lambd):
        self.lambd = lambd

    def eval_value(self, w, xis, yis):
        """ Result is a scalar. """
        w0 = w[0]
        w1 = w[1:]

        n = xis.shape[0]
        s = 0.0
        for i in range(n):
            s += max(0, 1 - yis[i] * (xis[i, :].dot(w1) - w0))
        return s / n + self.lambd*w1.dot(w1)

    def eval_grad(self, w, xis, yis):
        """ Result should be a vector of dimensionality of input w. """
        w0 = w[0]
        w1 = w[1:]

        dim = w.shape[0]
        n = xis.shape[0]
        s = np.zeros(dim)
        for i in range(n):
            # Note, like above, we handle the derivative of max() below, but we
            # must also handle w[0]' case specially.
            if yis[i]*(xis[i, :].dot(w1) - w0) < 1:
                s[0] += yis[i]*w0
                s[1:] += -yis[i] * xis[i, :]
            # else: s += 0, but that is not necessary.
        s /= n
        s[1:] += 2*self.lambd*w1
        return s
