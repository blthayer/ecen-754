import unittest
import project
from scipy.misc import derivative
from scipy.optimize import minimize, check_grad
import numpy as np

# Seed the random number generator for consistent results.
SEED = 42
np.random.seed(SEED)


class GradientTestCase(unittest.TestCase):
    """Ensure our gradient is behaving well."""

    def test_one(self):
        m = 100
        n = 200
        x = project.init_x(n, zeroes=False) * 0.01
        a = project.init_a(m, n, x)

        err = check_grad(project.objective, project.gradient, x, a)

        self.assertLess(err, 1e-4)

    def test_two(self):
        a = np.array([
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1]
        ])

        x = np.array([0.1, 0.2, 0.3])

        err = check_grad(project.objective, project.gradient, x, a)

        self.assertLess(err, 10e-8)


class SolverTestCase(unittest.TestCase):
    """Test our various solvers, comparing with scipy."""
    @classmethod
    def setUpClass(cls) -> None:
        # Lots and lots of initialization
        cls.m = 8
        cls.n = 10
        cls.alpha = 0.25
        cls.beta = 0.5
        cls.eta = 1e-6
        cls.eps = 1e-6
        cls.it_max = 1000

        cls.x_0 = project.init_x(cls.n, zeroes=True)
        cls.a = project.init_a(cls.m, cls.n, cls.x_0)

        # Test with scipy, using default/simple arguments.
        cls.result_simple = minimize(fun=project.objective, x0=cls.x_0,
                                     args=(cls.a,))

        # Test with scipy, using our gradient.
        cls.result_grad = minimize(fun=project.objective, jac=project.gradient,
                                   x0=cls.x_0, args=(cls.a,), method='BFGS',
                                   options={'gtol': cls.eta})

        # Test with scipy, using our hessian, too.
        cls.result_hess = minimize(fun=project.objective, jac=project.gradient,
                                   hess=project.hessian, x0=cls.x_0,
                                   args=(cls.a,), method='Newton-CG',
                                   tol=cls.eps)

    def test_gradient_descent(self):
        # Perform gradient descent.
        x, obj_list, t_list = project.gradient_descent(
            x=self.x_0, a=self.a, alpha=self.alpha, beta=self.beta,
            eta=self.eta, it_max=self.it_max)

        # Compare with gradient result.
        np.testing.assert_allclose(x, self.result_grad.x, rtol=1e-6, atol=0)

        # Compare with simple result.
        # Use a slightly higher tolerance since we're not specifying
        # the gradient tolerance.
        np.testing.assert_allclose(x, self.result_simple.x, rtol=5e-6, atol=0)

    def test_damped_newton(self):
        # Perform newton's method.
        x, obj_list, t_list = project.damped_newton(
            x=self.x_0, a=self.a, alpha=self.alpha, beta=self.beta,
            eps=self.eps, it_max=self.it_max)

        # Compare with simple result.
        # Use a slightly higher tolerance since we're not specifying
        # the gradient tolerance.
        np.testing.assert_allclose(x, self.result_simple.x, rtol=5e-6, atol=0)

        # Compare with newton-cg result.
        np.testing.assert_allclose(x, self.result_hess.x, rtol=1e-6, atol=0)

        pass


if __name__ == '__main__':
    unittest.main()

