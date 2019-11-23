"""
Course project for ECEN 754, Texas A&M University, Fall 2019
Author: Brandon Thayer
"""
import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator for consistent results.
SEED = 42
np.random.seed(SEED)


def main():
    # Initialize three problem instances.
    instances = []

    part_a()

    plt.show()


def part_a():
    """Do all the work for part a.
    """
    # Start with a reasonably sized problem.
    m = 8
    n = 10
    alpha = 0.25
    beta = 0.5
    eta = 1e-6
    it_max = 1000

    # Initialize our "x" vector to 0.
    x_0 = init_x(n, zeroes=True)

    # Initialize array to hold our "a" vectors in the columns.
    a = init_a(m, n, x_0)

    # Perform gradient descent.
    x, obj_array, t_list = gradient_descent(x=x_0, a=a, eta=eta, alpha=alpha,
                                            beta=beta, it_max=it_max)

    # Create listing of parameters.
    param_size = _get_param_size_str(m=m, n=n, alpha=alpha, beta=beta, eta=eta)

    # Plot.
    plot_results(obj_array=obj_array, t_list=t_list, method='Gradient Descent',
                 param_str = param_size)

    print('Done solving and plotting initial problem.')

    # # Determine the effect of alpha and beta for different problems.
    # alpha_array = np.arange(0.05, 0.5, 0.05)
    # beta_array = np.arange(0.1, 1, 0.1)
    # m_list = [8, 80, 800]
    # n_list = [10, 100, 1000]
    #
    # # Loop over problem sizes.
    # for m, n in zip(m_list, n_list):
    #     # Initialize problem.
    #     x_0 = init_x(n, zeroes=True)
    #     a = init_a(m, n, x_0)
    #
    #     # TODO: Initialize subplots here.
    #
    #     # Loop over backtracking parameters.
    #     for alpha in alpha_array:
    #         for beta in beta_array:
    #             gradient_descent(x=x_0, a=a, eta=eta, alpha=alpha,
    #                              beta=beta, it_max=it_max)
    #
    #     print(f'Done looping over alpha and beta for m={m}, n={n}')


def plot_results(obj_array, t_list, method, param_str):
    # TODO: Save figures.

    # Plot objective value vs. iterations.
    fig = plt.figure()
    ax = fig.gca()
    ax.semilogy(np.abs(obj_array))
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel(r'$|f(\mathbf{x})|$ (Log Scale)')
    ax.set_title('Objective Value (Log Scale) vs. Iterations\nMethod: '
                 + method + param_str)

    # Plot step size vs. iteration number.
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(t_list)
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel(r'Step Size, $t$')
    ax2.set_title('Step Size vs. Iteration Number\nMethod: ' + method
                  + param_str)

    # Plot the optimality gap.
    fig3 = plt.figure()
    ax3 = fig3.gca()
    ax3.semilogy(obj_array - obj_array[-1])
    ax3.set_xlabel('Iteration Number')
    ax3.set_ylabel(r'$f(\mathbf{x}) - p^*$ (Log Scale)')
    ax3.set_title('Optimality Gap vs. Iterations\nMethod: ' + method
                  + param_str)


def _get_param_size_str(m, n, alpha, beta, eta):
    """Helper for creating string of problem size and parameters for
    plot titles.
    """
    size_str = fr'Size: $m={m}, n={n}$'
    param_str = fr'Parameters: $\alpha={alpha}, \beta={beta}, \eta={eta}$'
    return '\n' + size_str + '\n' + param_str


def init_a(m, n, x, it_max=100):
    """Helper to initialize random a that results in a non-nan
    objective."""
    nan = True
    i = 0
    while nan and (i < it_max):
        a = np.random.random((m, n))
        obj = objective(x, a)
        nan = np.isnan(obj)

    if i >= it_max:
        raise UserWarning(f'Hit {i} iterations in init_a.')

    # noinspection PyUnboundLocalVariable
    return a


def init_x(n, zeroes=True):
    """Helper to initialize x to either 0's or random."""
    if zeroes:
        return np.zeros(n)
    else:
        return np.random.random(n)


def objective(x, a):
    """Evaluate the objective function for given x vector and a matrix."""
    return -np.log(1 - np.matmul(a, x)).sum() \
           - np.log(1 - np.square(x)).sum()


def gradient(x, a):
    """Evaluate the gradient of objective function for given x vector
    and A matrix.
    """
    # Compute b and one minus b
    b = np.matmul(a, x)
    one_m_b = 1 - b

    # Compute our x term.
    x_term = 2 * x / (1 - np.square(x))

    # Initialize our gradient.
    grad_x = np.zeros_like(x)

    # Loop over the size of grad_x.
    # TODO: I feel like this could be better vectorized, but I'm
    #   having some trouble.
    for k in range(len(grad_x)):
        # Add the correct x_term.
        grad_x[k] += x_term[k]

        # Add each component of the a / (1 - b) term.
        for i in range(len(b)):
            grad_x[k] += a[i, k] / one_m_b[i]

    return grad_x


def hessian(x, a):
    """Evaluate the Hessian matrix of the objective function for given x
        vector and A matrix.
    """
    # Start by getting the diagonal.
    h, one_m_b_2 = diag_hessian(x, a)

    # Fill in the lower triangle.
    for j in range(len(x)):
        for k in range(len(x)):
            # Skip the diagonal and upper triangle
            if k >= j:
                break

            # Loop over the rows in a.
            for i in range(a.shape[0]):
                h[j, k] += a[i, k] * a[i, j] / one_m_b_2[i]

    # Fill in the upper triangle.
    # Source: https://stackoverflow.com/a/2573982
    h = h + h.T - np.diag(h.diagonal())

    # Done.
    return h


def diag_hessian(x, a):
    """Evaluate only the diagonal of the Hessian matrix.
    """
    # Initialize.
    h = np.zeros((len(x), len(x)))

    # Pre-compute b, and (one minus b)^2.
    b = np.matmul(a, x)
    one_m_b_2 = np.square(1 - b)

    # Pre-compute the x terms that will go on the diagonal.
    x_sq = np.square(x)
    one_m_x_2 = 1 - x_sq
    x_term = 2 / one_m_x_2 + 4 * x_sq / np.square(one_m_x_2)

    # Pre-compute the square of the elements in a.
    a_sq = np.square(a)

    # Loop to fill in the diagonal
    # TODO: This also feels like it should be vectorized...
    for k in range(len(x)):
        # Loop to compute the summation term.
        for i in range(a.shape[0]):
            h[k, k] += a_sq[i, k] / one_m_b_2[i]

    return h, one_m_b_2


def gradient_descent(x, a, eta, alpha, beta, it_max):
    """Perform simple gradient descent with back-tracking line search.
    """

    # Get an initial gradient.
    g = gradient(x, a)

    # Compute the norm.
    norm = np.linalg.norm(g)

    # Initialize lists to track our objective values and step sizes.
    obj_list = []
    t_list = []

    # Loop while the norm is less than eta.
    i = 0
    while (eta <= norm) and (i < it_max):
        # Perform back-tracking line search to get our step size.
        t = backtrack_line_search(x=x, a=a, g=g, dx=-g, alpha=alpha, beta=beta)
        t_list.append(t)

        # Perform the x update.
        x = x - t * g

        # Compute new gradient and norm.
        g = gradient(x, a)
        norm = np.linalg.norm(g)

        # Compute new value of objective function, append to list.
        obj_list.append(objective(x, a))

        if np.isnan(obj_list[-1]):
            raise ValueError(
                'NaN objective value encountered in gradient_descent')

        # Update iteration counter.
        i += 1

    if i >= it_max:
        raise ValueError(f'Hit {i} iterations in gradient_descent.')

    return x, np.array(obj_list), t_list


def backtrack_line_search(x, a, g, dx, alpha, beta, it_max=1000):
    """Perform back-tracking line search to determine step length.

    :param x: Vector of x's
    :param a: Matrix of a_i column vectors that goes everywhere.
    :param g: Gradient(x)
    :param dx: Search direction. For gradient descent this will just
        be the gradient(x).
    :param alpha: 0 < alpha < 0.5
    :param beta: 0 < beta < 1
    :param it_max: Maximum number of iterations for the while loop.
    """
    # Evaluate f(x + t * dx) where t starts as one.
    t = 1
    new_obj = objective(x + t * dx, a)

    # Compute f(x)
    old_obj = objective(x, a)

    # Compute alpha * t * grad * dx
    g_dx = np.matmul(g, dx)
    term = alpha * t * g_dx

    # Initialize iteration counter.
    i = 0

    # Loop.
    while ((new_obj > old_obj + term) or np.isnan(new_obj)) and (i < it_max):
        # Update t.
        t *= beta

        # Re-compute our new objective value and "term"
        new_obj = objective(x + t * dx, a)
        term = alpha * t * g_dx

        i += 1

    if i >= it_max:
        raise UserWarning(f'Hit {i} iterations in backtrack_line_search.')

    # All done. Return the step size.
    return t


def damped_newton(x, a, eps, alpha, beta, it_max, hessian_update=1):
    """Use damped Newton's method to solve."""
    # Get initial gradient and Hessian.
    g = gradient(x, a)
    h = hessian(x, a)

    # Compute initial inverse of the Hessian.
    h_inv = np.linalg.inv(h)

    # Track how our objective value and step size changes over time.
    obj_list = []
    t_list = []

    # Initialize iteration counter.
    i = 0
    check = 0.5 * np.matmul(np.matmul(g, h_inv), g)
    while (check > eps) and (i < it_max):
        # Compute step direction.
        dx = -np.matmul(h_inv, g)

        # Get and track step size.
        t = backtrack_line_search(x=x, a=a, g=g, dx=dx, alpha=alpha, beta=beta)
        t_list.append(t)

        # Update x.
        x = x + t * dx

        # Possibly update the Hessian.
        if i % 1 == 0:
            h = hessian(x, a)
            h_inv = np.linalg.inv(h)

        # Update the gradient.
        g = gradient(x, a)

        # Track value of objective function.
        obj_list.append(objective(x, a))

        check = 0.5 * np.matmul(np.matmul(g, h_inv), g)

    if i >= it_max:
        raise UserWarning(f'Hit {i} iterations in backtrack_line_search.')

    return x, np.array(obj_list), t_list



if __name__ == '__main__':
    main()
