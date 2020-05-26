import unittest as ut
from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.tgedmd as tgedmd
import scikit_tt.data_driven.transform as tdt
from examples.lemon_slice import LemonSlice
from scikit_tt import TT


class TestHelperFunctions(TestCase):
    def setUp(self):
        self.tol = 1e-8

    def test_frobenius_inner(self):
        a = np.random.random((4, 4))
        b = np.random.random((4, 4))

        frob = 0
        for i in range(4):
            for j in range(4):
                frob += a[i, j] * b[i, j]

        self.assertLess(abs(frob - tgedmd._frob_inner(a, b)), self.tol)

    def test_generator(self):
        f = tdt.Sin(0, 0.5)
        x = np.array([1, 2, 3])

        def b(y):
            return np.array([y.sum(), 0, y.sum() ** 3])

        def sigma(y):
            return np.eye(3)

        generator = tgedmd._generator(f, x, b, sigma)
        generator2 = np.inner(f.gradient(x), b(x))
        generator2 += 0.5 * np.trace(f.hessian(x))

        self.assertLess(abs(generator - generator2), self.tol)


class TestCores(TestCase):
    def setUp(self):
        self.tol = 1e-8
        self.d = 4
        self.p = self.d

        self.ls = LemonSlice(k=4, beta=1, c=1, d=self.d, alpha=10)

        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Identity(i)] + [tdt.Monomial(i, j) for j in range(2, 6)])
        self.n = [len(mode) for mode in self.basis_list]

        self.x = np.random.random(self.d)
        self.a = self.ls.diffusion(self.x) @ self.ls.diffusion(self.x).T

        self.cores = [tgedmd.dPsix(self.basis_list[0], self.x, self.ls.drift, self.ls.diffusion, position='first')]
        self.cores = self.cores + [tgedmd.dPsix(self.basis_list[i], self.x, self.ls.drift, self.ls.diffusion,
                                                position='middle') for i in range(1, self.p - 1)]
        self.cores = self.cores + [tgedmd.dPsix(self.basis_list[-1], self.x, self.ls.drift, self.ls.diffusion,
                                                position='last')]

    def test_core0(self):
        core = self.cores[0]

        self.assertEqual(core.shape, (1, self.n[0], 1, self.d + 2))

        self.assertTrue((core[0, :, 0, 0] == np.array([fun(self.x) for fun in self.basis_list[0]])).all())
        self.assertTrue((core[0, :, 0, 1] == np.array([tgedmd._generator(fun, self.x, self.ls.drift, self.ls.diffusion)
                                                       for fun in self.basis_list[0]])).all())
        self.assertTrue((core[0, :, 0, 2] == np.array([fun.partial(self.x, 0) for fun in self.basis_list[0]])).all())
        self.assertTrue((core[0, :, 0, 3] == np.array([fun.partial(self.x, 1) for fun in self.basis_list[0]])).all())

    def test_core1(self):
        core = self.cores[1]

        self.assertEqual(core.shape, (self.d + 2, self.n[1], 1, self.d + 2))

        self.assertTrue((core[0, :, 0, 0] == np.array([fun(self.x) for fun in self.basis_list[1]])).all())
        self.assertTrue((core[2, :, 0, 2] == np.array([fun(self.x) for fun in self.basis_list[1]])).all())

        self.assertTrue((core[0, :, 0, 1] == np.array([tgedmd._generator(fun, self.x, self.ls.drift, self.ls.diffusion)
                                                       for fun in self.basis_list[1]])).all())
        self.assertTrue((core[0, :, 0, 2] == np.array([fun.partial(self.x, 0) for fun in self.basis_list[1]])).all())
        self.assertTrue((core[0, :, 0, 3] == np.array([fun.partial(self.x, 1) for fun in self.basis_list[1]])).all())

        self.assertTrue((core[3, :, 0, 1] == np.array([np.inner(self.a[1, :], fun.gradient(self.x))
                                                       for fun in self.basis_list[1]])).all())

    def test_core_last(self):
        core = self.cores[-1]

        self.assertEqual(core.shape, (self.d + 2, self.n[-1], 1, 1))

        self.assertTrue((core[0, :, 0, 0] == np.array([tgedmd._generator(fun, self.x, self.ls.drift, self.ls.diffusion)
                                                       for fun in self.basis_list[-1]])).all())
        self.assertTrue((core[1, :, 0, 0] == np.array([fun(self.x) for fun in self.basis_list[-1]])).all())

        self.assertTrue((core[2, :, 0, 0] == np.array([np.inner(self.a[0, :], fun.gradient(self.x))
                                                       for fun in self.basis_list[-1]])).all())
        self.assertTrue((core[-1, :, 0, 0] == np.array([np.inner(self.a[-1, :], fun.gradient(self.x))
                                                       for fun in self.basis_list[-1]])).all())

    def test_tensor(self):
        """
        Check if the full tensor dPsi(x) is correct.
        """
        tensor = TT(self.cores)
        tensor = np.squeeze(tensor.full())

        # create reference tensor from scratch
        t_ref = np.zeros(self.n)
        for s0 in range(self.n[0]):
            for s1 in range(self.n[1]):
                for s2 in range(self.n[2]):
                    for s3 in range(self.n[3]):
                        t_ref[s0, s1, s2, s3] = tgedmd.generator_on_product(self.basis_list, (s0, s1, s2, s3), self.x,
                                                                            self.ls.drift, self.ls.diffusion)
        self.assertTrue((np.abs(tensor - t_ref) < self.tol).all())


class TestTTDecomposition(TestCase):
    def setUp(self):
        self.tol = 1e-8
        self.d = 4
        self.p = self.d
        self.m = 6

        self.ls = LemonSlice(k=4, beta=1, c=1, d=self.d, alpha=10)

        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Identity(i)] + [tdt.Monomial(i, j) for j in range(2, 6)])
        self.n = [len(mode) for mode in self.basis_list]

        self.x = np.random.random((self.d, self.m))
        self.a = self.ls.diffusion(self.x) @ self.ls.diffusion(self.x).T

    def test_tt_decomposition(self):
        dPsiX = tgedmd.tt_decomposition(self.x, self.basis_list, self.ls.drift, self.ls.diffusion)
        dPsiX = np.squeeze(dPsiX.full())

        # construct reference
        t_ref = np.zeros(list(self.n) + [self.m])

        for k in range(self.m):
            for s0 in range(self.n[0]):
                for s1 in range(self.n[1]):
                    for s2 in range(self.n[2]):
                        for s3 in range(self.n[3]):
                            t_ref[s0, s1, s2, s3, k] = tgedmd.generator_on_product(self.basis_list, (s0, s1, s2, s3),
                                                                                   self.x[:, k],
                                                                                   self.ls.drift, self.ls.diffusion)

        self.assertTrue((np.abs(dPsiX - t_ref) < self.tol).all())


if __name__ == '__main__':
    ut.main()
