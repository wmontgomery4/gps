""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics

from IPython.core.debugger import Tracer; debug_here = Tracer()

class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = \
                self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    #TODO: Merge this with DynamicsLR.fit - lots of duplicated code.
    def fit(self, sample_list, traj_distr=None, traj_info=None):
        """ Fit dynamics. """
        X = sample_list.get_X()  # Use all samples to fit dynamics.
        U = sample_list.get_U()
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        ix = slice(dX)
        iu = slice(dX,dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)

        # Track analytical state-action mu/sigma.
        if traj_distr is not None:
            assert traj_info is not None
            xu_sigma = np.zeros((T, dX+dU, dX+dU))
            xu_mu = np.zeros((T, dX+dU))

            # Set initial covariance (initial mu is always zero).
            xu_sigma[0, ix, ix] = traj_info.x0sigma
            xu_mu[0, ix] = traj_info.x0mu

        # Fit dynamics with least squares regression.
        for t in range(T - 1):
            # Evaluate prior.
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            mu0, Phi, m, n0 = self.prior.eval(dX, dU, xux)

            # Linear regression with prior.
            xux_mean = np.mean(xux, axis=0)
            empsig = (xux - xux_mean).T.dot(xux - xux_mean) / N
            empsig = 0.5 * (empsig + empsig.T)

            sigma = (N * empsig + Phi + (N * m) / (N + m) *
                     np.outer(xux_mean - mu0, xux_mean - mu0)) / (N + n0)
            sigma = 0.5 * (sigma + sigma.T)
            sigma[it, it] += self._hyperparams['regularization'] * np.eye(dX+dU)

            if traj_distr is not None:
                # Take analytical state-action step (policy)
                K, k = traj_distr.K[t, :, :], traj_distr.k[t, :]
                xu_sigma[t, :, :] = np.vstack([
                    np.hstack([
                        xu_sigma[t, ix, ix],
                        xu_sigma[t, ix, ix].dot(K.T)
                    ]),
                    np.hstack([
                        K.dot(xu_sigma[t, ix, ix]),
                        K.dot(xu_sigma[t, ix, ix]).dot(K.T) +
                                traj_distr.pol_covar[t, :, :]
                    ])
                ])
                xu_mu[t, :] = np.hstack([
                    xu_mu[t, ix],
                    xux_mean[iu]
#                    K.dot(xu_mu[t, ix]) + k
                ])

            # Condition on dynamics.
            Fm = np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip]).T
            if traj_distr is not None: # Empirical mean matching
                fv = xux_mean[ip] - Fm.dot(xu_mu[t, it])
            else:
                fv = xux_mean[ip] - Fm.dot(xux_mean[it])

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5 * (dyn_covar + dyn_covar.T)

            # Take analytical state-action step (dynamics)
            if (traj_distr is not None) and (t < T - 1):
                xu_sigma[t+1, ix, ix] = self.dyn_covar[t, :, :] + \
                        Fm.dot(xu_sigma[t, :, :]).dot(Fm.T)
                xu_mu[t+1, ix] = Fm.dot(xu_mu[t, :]) + fv
