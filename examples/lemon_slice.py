import numpy as np

from scipy.integrate import quad, nquad

""" Implements 2d-Lemon Slice potential as model system: """


class LemonSlice:
    """
        Stochastic dynamics in two-dimensional Lemon Slice potential, given by:

        V(x, y) = cos(k*phi) + 1.0/np.cos(0.5*phi) + 10*(r-1)**2 + (1.0/r)

        where r, phi are the polar coordinates of x, y.

        k, int:
            number of minima along the polar axis.
        beta, float:
            inverse temperature
        c, float:
            scaling factor for cosine part of the potential.
        d, int (optional):
            dimension of the system. If not equal to two, the potential
            will just be harmonic along all additional dimensions.
    """

    def __init__(self, k, beta, c=1.0, d=2, alpha=1.0):
        self.k = k
        self.beta = beta
        self.c = c
        self.d = d
        self.alpha = alpha

        # Compute partition function:
        # start with "Lemon Slice" part:
        dims = np.array([[-2.51, 2.51],
                         [-2.51, 2.51]])
        Z = nquad(lambda x, y: np.exp(-self.beta * self._V_lemon_slice(x, y)), dims)[0]
        # Multiply by partition functions of the Gaussians along all other directions:
        for ii in range(2, self.d):
            Z *= np.sqrt(self.beta * self.alpha / (2 * np.pi))
        self.Z = Z

        # Calculate normalization constants for effective dynamics:
        self.C1 = quad(lambda r: np.exp(-(10 * (r - 1) ** 2 + (1.0 / r))) * (1.0 / r), 0, 6.0)[0]
        self.C2 = quad(lambda r: np.exp(-(10 * (r - 1) ** 2 + (1.0 / r))) * r, 0, 6.0)[0]

    def Simulate(self, x0, m, dt):
        """
            Generate trajectory of stochastic dynamics in Lemon Slice potential, using Euler scheme.

        x0, nd-array(d):
            initial values for the simulation.
        m, int:
            number time steps to be returned (including initial value)
        dt, float:
            integration time step

        Returns:
        --------
        X, nd-array (d, m):
            simulation trajectory
        """
        # Initialize:
        X = np.zeros((self.d, m))
        X_old = x0[:, None]
        # print(X_old)
        X[:, 0] = X_old[:, 0]
        # Run simulation:
        for t in range(1, m):
            # print("t = %d:"%t)
            # print(self.gradient(X_old))
            X_new = X_old - self.gradient(X_old) * dt + \
                    np.sqrt(2 * dt / self.beta) * np.random.randn(self.d, 1)
            # print(X_new)
            X[:, t] = X_new[:, 0]
            X_old = X_new
            # print("")
        return X

    def potential(self, x):
        """
            Evaluate potential energy at Euclidean positions x

            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            V, nd-array (m,):
                Values of the potential for all pairs of x-y-values.
        """
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the potential:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        V = self.c * np.cos(self.k * phi) + 1.0 / np.cos(0.5 * phi) + 10 * (r - 1) ** 2 + (1.0 / r)
        # Add harmonic terms for all remaining dimensions:
        for ii in range(2, self.d):
            V += 0.5 * self.alpha * x[ii, :] ** 2

        return V

    def gradient(self, x):
        """
            Evaluate gradient of potential energy at Euclidean positions x

            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            dx, nd-array (d, m)
                Gradient of the potential for all m data points in x.
        """
        dV = np.zeros((self.d, x.shape[1]))
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the gradient:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        dV[0, :] = -(0.5 * np.sin(0.5 * phi) / np.cos(0.5 * phi) ** 2 -
                     self.c * self.k * np.sin(self.k * phi)) * (x[1, :] / r ** 2) + 20 * (r - 1) * (x[0, :] / r) - (
                               1.0 / r ** 2) * (x[0, :] / r)
        dV[1, :] = (0.5 * np.sin(0.5 * phi) / np.cos(0.5 * phi) ** 2 -
                    self.c * self.k * np.sin(self.k * phi)) * (x[0, :] / r ** 2) + 20 * (r - 1) * (x[1, :] / r) - (
                               1.0 / r ** 2) * (x[1, :] / r)
        # Add harmonic contributions to all remaining dimensions:
        for ii in range(2, self.d):
            dV[ii, :] = self.alpha * x[ii, :]
        return dV

    def drift(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            single point, shape (d,)

        Returns
        -------
        np.ndarray
        """
        return -self.gradient(x[:, np.newaxis])[:, 0]

    def diffusion(self, x):
        """
        Evaluate diffusion sigma at position x
        """
        return (2.0 / self.beta) ** 0.5 * np.eye(self.d)

    def stat_dist(self, x):
        """
            Evaluate stationary density at Euclidean positions x

            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            mu, nd-array (m):
                Values of the stationary density for all m data points in x.
        """
        return (1.0 / self.Z) * np.exp(-self.beta * self.potential(x))

    def _V_lemon_slice(self, x, y):
        """ Return only the "Lemon Slice" part of the potential."""
        r, phi = self._polar_rep(x, y)
        return self.c * np.cos(self.k * phi) + 1.0 / np.cos(0.5 * phi) + 10 * (r - 1) ** 2 + (1.0 / r)

    @staticmethod
    def _polar_rep(x, y):
        """
            Compute polar coordinates from 2d Euclidean coordinates:

            x, y, nd-array (m):
                Arrays of two-dimensional Euclidean coordinates to be transformed.

            Returns:
            --------
            r, phi, nd-array (m):
                Arrays of polar coordinates corresponding to x and y.
        """
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return r, phi
