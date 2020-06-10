import unittest as ut
from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.tgedmd as tgedmd
import scikit_tt.data_driven.transform as tdt
from examples.lemon_slice import LemonSlice
from scikit_tt.tensor_train import TT


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

    def test_special_tensordot(self):
        dimsA = [(10, 10, 3, 4), (1, 10, 3, 4), (10, 10, 3, 4), (1, 10, 3, 4)]
        dimsB = [(10, 10, 4, 3), (10, 10, 4, 3), (10, 1, 4, 3), (10, 1, 4, 3)]

        for idx in range(len(dimsA)):
            dimAi, dimAj, dimA1, dimA2 = dimsA[idx]
            dimBi, dimBj, dimB1, dimB2 = dimsB[idx]
            A = np.zeros((dimAi, dimAj, dimA1, dimA2))
            B = np.zeros((dimBi, dimBj, dimB1, dimB2))

            if A.shape[0] == 1:
                A = np.random.random(A.shape)
            else:
                # diagonal
                for i in range(min(A.shape[0], A.shape[1])):
                    A[i, i, :, :] = np.random.random((dimA1, dimA2))
                # first row
                A[0, :, :, :] = np.random.random((dimAj, dimA1, dimA2))
                # second column
                A[:, 1, :, :] = np.random.random((dimAi, dimA1, dimA2))

            if B.shape[1] == 1:
                B = np.random.random(B.shape)
            else:
                # diagonal
                for i in range(min(B.shape[0], B.shape[1])):
                    B[i, i, :, :] = np.random.random((dimB1, dimB2))
                # first row
                B[0, :, :, :] = np.random.random((dimBj, dimB1, dimB2))
                # second column
                B[:, 1, :, :] = np.random.random((dimBi, dimB1, dimB2))

            C1 = np.tensordot(A, B, axes=((1, 3), (0, 2)))
            C1 = np.transpose(C1, [0, 2, 1, 3])
            C2 = tgedmd._special_tensordot(A, B)

            self.assertTrue((np.abs(C1 - C2) < self.tol).all())


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

    def test_tt_decomposition_chunks(self):
        dPsiX = tgedmd.tt_decomposition(self.x, self.basis_list, self.ls.drift, self.ls.diffusion)
        dPsiX_chunks = tgedmd.tt_decomposition_chunks(self.x, self.basis_list, self.ls.drift, self.ls.diffusion,
                                                      threshold=0, chunk_size=2)

        dPsiX = np.squeeze(dPsiX.full())
        dPsiX_chunks = np.squeeze(dPsiX_chunks.full())

        self.assertTrue((np.abs(dPsiX - dPsiX_chunks) < self.tol).all())


class TestAMUSEt(TestCase):
    def setUp(self):
        self.tol = 1e-8
        self.d = 4
        self.p = self.d
        self.m = 6

        self.ls = LemonSlice(k=4, beta=1, c=1, d=self.d, alpha=10)

        self.basis_list = []
        for i in range(self.d):
            self.basis_list.append([tdt.Identity(i)] + [tdt.Monomial(i, j) for j in range(2, 6)])

        self.x = np.random.random((self.d, self.m))

        # build psi and dpsi
        self.psi = tdt.basis_decomposition(self.x, self.basis_list)
        p = self.psi.order - 1
        self.u, self.s, self.v = self.psi.svd(p)
        self.s_inv = np.diag(1.0 / self.s)
        self.us = self.u.rank_tensordot(self.s_inv, mode='last')

        self.dpsi = tgedmd.tt_decomposition(self.x, self.basis_list, self.ls.drift, self.ls.diffusion)

    def test_amuset_chunks(self):
        # with standard tensordot
        M = tgedmd._amuset(self.us, self.v, self.dpsi)
        M2 = tgedmd._amuset_chunks(self.u, self.s, self.v, self.x, self.basis_list, self.ls.drift, self.ls.diffusion,
                                   threshold=0, max_rank=np.infty, chunk_size=2)

        self.assertTrue((np.abs(M - M2) < self.tol).all())

    def test_contract_dPsi_u(self):
        # only works for chunk_size=1 (or m=1)
        m = 1
        x = np.random.random((self.d, m))
        psi = tdt.basis_decomposition(x, self.basis_list)
        p = psi.order - 1
        u, s, v = psi.svd(p)
        s_inv = np.diag(1.0 / s)
        us = u.rank_tensordot(s_inv, mode='last')
        dpsi = tgedmd.tt_decomposition(x, self.basis_list, self.ls.drift, self.ls.diffusion)

        # calculate M using normal tensordot
        M = tgedmd._amuset(us, v, dpsi)

        # calculate M using special tensordot
        M2 = tgedmd._amuset_special(us, v, dpsi)

        self.assertTrue((np.abs(M - M2) < self.tol).all())

    def test_amuset_chunks_special(self):
        # with special tensordot (chunk_size = 1)
        M = tgedmd._amuset(self.us, self.v, self.dpsi)
        M2 = tgedmd._amuset_chunks(self.u, self.s, self.v, self.x, self.basis_list, self.ls.drift, self.ls.diffusion,
                                   threshold=0, max_rank=np.infty, chunk_size=1)

        self.assertTrue((np.abs(M - M2) < self.tol).all())


if __name__ == '__main__':
    ut.main()
