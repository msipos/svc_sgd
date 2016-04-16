import numpy as np


class GradientDescentSolver:
    """ This is a plain gradient descent solver, no stochasticity. """
    def __init__(self, eta, N, loss_function):
        self.eta = eta
        self.N = N
        self.lf = loss_function

    def solve(self, start_w):
        losses = []
        ws = []

        w = np.copy(start_w)

        # Evaluate for start_w:
        losses.append(self.lf.eval_value(w))
        ws.append(np.copy(w))

        for step in range(self.N):
            grad = self.lf.eval_grad(w)
            w += -self.eta*grad
            loss = self.lf.eval_value(w)

            # Keep track of these:
            losses.append(loss)
            ws.append(np.copy(w))

        return losses, ws


class StochasticGradientDescentSolver:
    """ Stochastic version of the above. """
    def __init__(self, loss_function, xis, yis):
        self.lf = loss_function
        self.xis = np.copy(xis)
        self.yis = np.copy(yis)
        self.num_data_points = self.xis.shape[0]

    def _shuffle(self):
        order = np.arange(self.num_data_points)
        np.random.shuffle(order)
        xis = self.xis[order,:]
        yis = self.yis[order]
        return xis, yis

    def solve(self, start_w, eta, N, batch_size):
        losses = []
        ws = []

        w = np.copy(start_w)

        # Evaluate for start_w:
        losses.append(self.lf.eval_value(w, self.xis, self.yis))
        ws.append(np.copy(w))

        # NOTE: Gradients are evaluated with lf, whereas the total loss
        # is evaluated via total_lf.
        for step in range(N):
            xis, yis = self._shuffle()

            for start_batch in range(0, self.num_data_points, batch_size):
                end_batch = start_batch + batch_size
                if end_batch > self.num_data_points:
                    end_batch = self.num_data_points
                b_xis = xis[start_batch:end_batch, :]
                b_yis = yis[start_batch:end_batch]

                grad = self.lf.eval_grad(w, b_xis, b_yis)
                w += -eta*grad

            loss = self.lf.eval_value(w, self.xis, self.yis)

            # Keep track of these:
            losses.append(loss)
            ws.append(np.copy(w))

            if len(losses) > 1:
                prev_loss = losses[-2]
                # Break out of the loop when loss improvement rate is down to 0.1%
                if loss / prev_loss > 0.999 and loss / prev_loss < 1.001:
                    break

        return losses, ws
