# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import scikit_tt.data_driven.transform as tdt


class TestMANDy(TestCase):

    def setUp(self):
        """..."""

        self.tol = 1e-10
        self.d = 10
        self.m = 5
        self.n = 20
        self.data = np.random.rand(self.d, self.m)
        self.data_2 = np.random.rand(self.d, self.n)
        self.phi_1 = [[tdt.ConstantFunction(), tdt.Identity(i), tdt.Monomial(i, 2)] for i in range(self.d)]
        self.psi_1 = [lambda t: 1, lambda t: t, lambda t: t**2]
        self.phi_2 = [[tdt.ConstantFunction()] + [tdt.Sin(i, 1) for i in range(self.d)], [tdt.ConstantFunction()] + [tdt.Cos(i, 1) for i in range(self.d)]]
        self.psi_2 = [lambda t: np.sin(t), lambda t: np.cos(t)]


    def test_basis_functions(self):
        """test basis functions"""

        constant = tdt.ConstantFunction()
        indicator = tdt.IndicatorFunction(0, 0, 0.5)
        identity = tdt.Identity(0)
        monomial = tdt.Monomial(0, 2)
        sin = tdt.Sin(0, 1)
        cos = tdt.Cos(0, 1)
        gauss = tdt.GaussFunction(0, 1, 1)
        periodic_gauss = tdt.PeriodicGaussFunction(0, 1, 1)
        
        self.assertEqual(np.sum(np.abs(constant(self.data)-np.ones(self.m))), 0)
        self.assertEqual(np.sum(np.abs(indicator(self.data)-np.logical_and(self.data[0, :]>=0, self.data[0, :]<0.5))), 0)
        self.assertEqual(np.sum(np.abs(identity(self.data)-self.data[0, :])), 0)
        self.assertEqual(np.sum(np.abs(monomial(self.data)-self.data[0, :]**2)), 0)
        self.assertEqual(np.sum(np.abs(sin(self.data)-np.sin(self.data[0,:]))), 0)
        self.assertEqual(np.sum(np.abs(cos(self.data)-np.cos(self.data[0,:]))), 0)
        self.assertEqual(np.sum(np.abs(gauss(self.data)-np.exp(-0.5 * (self.data[0,:] - 1) ** 2))), 0)
        self.assertEqual(np.sum(np.abs(periodic_gauss(self.data)-np.exp(-0.5 * np.sin(0.5 * (self.data[0,:] - 1)) ** 2))), 0)
        
    def test_basis_decomposition(self):
        """test construction of transformed data tensors"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = np.zeros([3**self.d, self.m])
        for j in range(self.m):
            v = [1, self.data[0,j], self.data[0,j]**2]
            for i in range(1,self.d):
                v = np.kron(v, [1, self.data[i,j], self.data[i,j]**2])
            tdt_2[:,j] = v
        self.assertEqual(np.sum(np.abs(tdt_1-tdt_2)), 0)

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1)
        core_0 = tdt.basis_decomposition(self.data, self.phi_1, single_core=0)
        core_1 = tdt.basis_decomposition(self.data, self.phi_1, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0]-core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1]-core_1)), 0)

    def test_coordinate_major(self):
        """test coordinate-major decomposition"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1)
        tdt_2 = tdt.coordinate_major(self.data, self.psi_1)
        self.assertLess((tdt_1-tdt_2).norm(), self.tol)

        core_0 = tdt.coordinate_major(self.data, self.psi_1, single_core=0)
        core_1 = tdt.coordinate_major(self.data, self.psi_1, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0]-core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1]-core_1)), 0)

    def test_function_major(self):
        """test function-major decomposition"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_2)
        _ = tdt.function_major(self.data, self.psi_2, add_one=False)
        _ = tdt.function_major(self.data, self.psi_2, add_one=False, single_core=0)
        _ = tdt.function_major(self.data, self.psi_2, add_one=False, single_core=1)
        tdt_2 = tdt.function_major(self.data, self.psi_2)
        self.assertLess((tdt_1-tdt_2).norm(), self.tol)

        core_0 = tdt.function_major(self.data, self.psi_2, single_core=0)
        core_1 = tdt.function_major(self.data, self.psi_2, single_core=1)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[0]-core_0)), 0)
        self.assertEqual(np.sum(np.abs(tdt_1.cores[1]-core_1)), 0)

    def test_gram(self):
        """test construction of gram matrix"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = tdt.basis_decomposition(self.data_2, self.phi_1).transpose(cores=[self.d]).matricize()
        gram = tdt.gram(self.data, self.data_2, self.phi_1)
        self.assertLess(np.sum(np.abs(tdt_1.T.dot(tdt_2)-gram)), self.tol)

    def test_hocur(self):
        """test higher-order CUR decomposition"""

        tdt_1 = tdt.basis_decomposition(self.data, self.phi_1).transpose(cores=[self.d]).matricize()
        tdt_2 = tdt.hocur(self.data, self.phi_1, 5, repeats=10, progress=False).transpose(cores=[self.d]).matricize()
        self.assertLess(np.sum(np.abs(tdt_1-tdt_2)), self.tol)








        
        