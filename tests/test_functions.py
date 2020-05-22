import unittest as ut
from unittest import TestCase
import scikit_tt.data_driven.transform as tdt
import numpy as np


class TestFunction(TestCase):
    def test_initialized(self):
        f = tdt.Function(3)

        self.assertEqual(f([1, 2, 3]), 0)
        with self.assertRaises(ValueError):
            f([1, 2])
        with self.assertRaises(ValueError):
            f([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            tdt.Function(0)

    def test_unitialized(self):
        f = tdt.Function()
        x = np.random.random((4,))

        self.assertEqual(f(x), 0)
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertTrue((f.gradient(x) - np.zeros((4,)) == 0).all())
        self.assertTrue((f.hessian(x) - np.zeros((4, 4)) == 0).all())

        with self.assertRaises(ValueError):
            f([1, 2])
        with self.assertRaises(ValueError):
            f([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            f.partial(x, -1)
        with self.assertRaises(ValueError):
            f.partial2(x, 0, 7)


class TestOneCoordinateFunction(TestCase):
    def test_initialized(self):
        f = tdt.OneCoordinateFunction(1, 3)

        self.assertEqual(f([1, 2, 3]), 0)
        with self.assertRaises(ValueError):
            f([1, 2])
        with self.assertRaises(ValueError):
            f([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            tdt.OneCoordinateFunction(4, 3)
        with self.assertRaises(ValueError):
            tdt.OneCoordinateFunction(-1, 3)

    def test_unitialized(self):
        f = tdt.OneCoordinateFunction(1)
        x = np.random.random((4,))

        self.assertEqual(f(x), 0)
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertTrue((f.gradient(x) - np.zeros((4,)) == 0).all())
        self.assertTrue((f.hessian(x) - np.zeros((4, 4)) == 0).all())

        with self.assertRaises(ValueError):
            f([1, 2])
        with self.assertRaises(ValueError):
            f([1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            f.partial(x, -1)
        with self.assertRaises(ValueError):
            f.partial2(x, 0, 7)


class TestConstantFunction(TestCase):
    def test_constant_function(self):
        f = tdt.ConstantFunction()
        x = np.random.random((3,))
        grad = np.zeros((3,))
        hess = np.zeros((3, 3))

        self.assertEqual(f(x), 1)
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 0)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), 0)
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())


class TestIndicatorFunction(TestCase):
    def test_indicator_function(self):
        f = tdt.IndicatorFunction(1, 0.5, 1.0)

        self.assertEqual(f([0, 0.75]), 1)
        self.assertEqual(f([0, -2]), 0)
        self.assertEqual(f([0.75, 2]), 0)

        with self.assertRaises(NotImplementedError):
            f.partial([0, 0.75], 0)
        with self.assertRaises(NotImplementedError):
            f.partial2([0, 0.75], 0, 0)
        with self.assertRaises(NotImplementedError):
            f.gradient([0, 0.75])
        with self.assertRaises(NotImplementedError):
            f.hessian([0, 0.75])


class TestIdentityFunction(TestCase):
    def test_identity_function(self):
        f = tdt.Identity(1)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 1
        hess = np.zeros((3, 3))

        self.assertEqual(f(x), x[1])
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 1)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), 0)
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())


class TestMonomial(TestCase):
    def test_monomial(self):
        f = tdt.Monomial(1, 3)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 3 * x[1] ** 2
        hess = np.zeros((3, 3))
        hess[1, 1] = 6 * x[1]

        self.assertEqual(f(x), x[1] ** 3)
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 3 * x[1] ** 2)
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), 6 * x[1])
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())

    def test_exception(self):
        with self.assertRaises(ValueError):
            f = tdt.Monomial(1, -4)

    def test_small_exponent(self):
        f = tdt.Monomial(1, 1)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 1
        hess = np.zeros((3, 3))

        self.assertEqual(f(x), x[1])
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())


class TestSin(TestCase):
    def test_sin(self):
        f = tdt.Sin(1, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = 0.5 * np.cos(0.5 * x[1])
        hess = np.zeros((3, 3))
        hess[1, 1] = -(0.5 ** 2) * np.sin(0.5 * x[1])

        self.assertEqual(f(x), np.sin(0.5 * x[1]))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), 0.5 * np.cos(0.5 * x[1]))
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), -(0.5 ** 2) * np.sin(0.5 * x[1]))
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())


class TestCos(TestCase):
    def test_sin(self):
        f = tdt.Cos(1, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = -0.5 * np.sin(0.5 * x[1])
        hess = np.zeros((3, 3))
        hess[1, 1] = -(0.5 ** 2) * np.cos(0.5 * x[1])

        self.assertEqual(f(x), np.cos(0.5 * x[1]))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), -0.5 * np.sin(0.5 * x[1]))
        self.assertEqual(f.partial2(x, 0, 0), 0)
        self.assertEqual(f.partial2(x, 1, 1), -(0.5 ** 2) * np.cos(0.5 * x[1]))
        self.assertTrue((f.gradient(x) - grad == 0).all())
        self.assertTrue((f.hessian(x) - hess == 0).all())


class TestGaussFunction(TestCase):
    def test_gauss_function(self):
        f = tdt.GaussFunction(1, 0.5, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = -np.exp(-(0.5 * (0.5 - x[1]) ** 2) / 0.5) * (-0.5 + x[1]) / 0.5

        self.assertEqual(f(x), np.exp(-0.5 * (x[1] - 0.5) ** 2 / 0.5))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), -np.exp(-(0.5 * (0.5 - x[1]) ** 2) / 0.5) * (-0.5 + x[1]) / 0.5)
        self.assertTrue((f.gradient(x) - grad == 0).all())

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            tdt.GaussFunction(1, 0.5, 0)


class TestPeriodicGaussFunction(TestCase):
    def test_gauss_function(self):
        f = tdt.PeriodicGaussFunction(1, 0.5, 0.5)
        x = np.random.random((3,))
        grad = np.zeros((3,))
        grad[1] = (0.5 * np.exp(-(0.5 * np.sin(0.5 * 0.5 - 0.5 * x[1]) ** 2) / 0.5) *
                   np.cos(0.5 * 0.5 - 0.5 * x[1]) * np.sin(0.5 * 0.5 - 0.5 * x[1])) / 0.5

        self.assertEqual(f(x), np.exp(-0.5 * np.sin(0.5 * (x[1] - 0.5)) ** 2 / 0.5))
        self.assertEqual(f.partial(x, 0), 0)
        self.assertEqual(f.partial(x, 1), (0.5 * np.exp(-(0.5 * np.sin(0.5 * 0.5 - 0.5 * x[1]) ** 2) / 0.5) *
                                           np.cos(0.5 * 0.5 - 0.5 * x[1]) * np.sin(0.5 * 0.5 - 0.5 * x[1])) / 0.5)
        self.assertTrue((f.gradient(x) - grad == 0).all())

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            tdt.GaussFunction(1, 0.5, 0)


if __name__ == '__main__':
    ut.main()
