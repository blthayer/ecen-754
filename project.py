"""
Course project for ECEN 754, Texas A&M University, Fall 2019
Author: Brandon Thayer
"""
import numpy as np
import matplotlib.pyplot as plt
# We'll use scipy ONLY to compute p* to a high accuracy to reduce code
# run time.
from scipy.optimize import minimize

# Seed the random number generator for consistent results.
SEED = 42
np.random.seed(SEED)


def main():
    """Main function for the project, which initializes problem
    instances which are re-used for each project component, and then
    calls the functions which solve the problem using our various
    methods.
    """
    # Initialize four problem instances with different sizes.
    instances = [
        {
            'm': 8, 'n': 10, 'alpha': 0.25, 'beta': 0.5, 'eta': 1e-6,
            'it_max': 100000, 'eps': 1e-12, 'gtol': 1e-9
        },
        {
            'm': 40, 'n': 50, 'alpha': 0.25, 'beta': 0.5, 'eta': 5e-6,
            'it_max': 1000000, 'eps': 1e-12, 'gtol': 1e-8
        },
        {
            'm': 80, 'n': 100, 'alpha': 0.25, 'beta': 0.5, 'eta': 1e-5,
            'it_max': 1000000, 'eps': 1e-12, 'gtol': 2e-8
        },
        {
            'm': 160, 'n': 200, 'alpha': 0.25, 'beta': 0.5, 'eta': 1e-4,
            'it_max': 10000000, 'eps': 1e-12, 'gtol': 1e-7
        }
    ]

    # Initialize x vectors and A matrices for each problem instance.
    # Also get strings for plotting, and p* values for plotting.
    for d in instances:
        d['x_0'] = init_x(n=d['n'], zeroes=True)
        d['a'] = init_a(m=d['m'], n=d['n'], x=d['x_0'], it_max=100)
        # noinspection PyTypeChecker
        d['param_str_eta'] = _get_param_size_str(
            m=d['m'], n=d['n'], alpha=d['alpha'], beta=d['beta'], eta=d['eta'])

        # noinspection PyTypeChecker
        d['param_str_eps'] = _get_param_size_str(
            m=d['m'], n=d['n'], alpha=d['alpha'], beta=d['beta'], eps=d['eps'])

        # Solve the problem to get p* for plotting purposes.
        # noinspection PyUnresolvedReferences,PyTypeChecker
        result = minimize(fun=objective, x0=d['x_0'].copy(),
                          args=(d['a'],), method='trust-exact',
                          jac=gradient, hess=hessian,
                          options={'gtol': d['gtol']})

        # Put p* in the dictionary.
        d['p*'] = result.fun

        print(f"Solved via scipy for m={d['m']}, n={d['n']}")

    # Perform all the project work.
    part_a(instances)
    part_b_and_c(instances)


def part_a(instances):
    """Do all the work for part a: use gradient descent to solve, plot
    objective value vs. iterations (log scale), step length vs.
    iterations, and f(x) - p* vs. iterations (log scale).

    Also, experiment with different alpha and beta values to see their
    effect on total iterations required for all three problem instances.
    """
    for d in instances:
        # Perform gradient descent with the first problem instance.
        x, obj_array, t_list = gradient_descent(**d)

        # Plot.
        plot_results(obj_array=obj_array, t_list=t_list,
                     method='Gradient Descent', param_str=d['param_str_eta'],
                     method_file='grad_desc', m=d['m'], n=d['n'], p_s=d['p*'])

        print(f"Initial problem solved (grad desc). m={d['m']}, n={d['n']}")

    # Determine the effect of alpha and beta for different problems.
    alpha_array = np.arange(0.05, 0.5, 0.05)
    beta_array = np.arange(0.1, 1, 0.1)

    # Loop over problem sizes.
    for d in instances:
        # Initialize suplots. Do 3x3 since alpha & beta have len 9.
        # We'll make the size fit with 0.5" margins, with an extra 0.5"
        # for safety.
        # noinspection PyTypeChecker
        fig, ax_it = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True,
                                  figsize=(9.5, 7))
        fig.suptitle(
            r'Number of Iterations vs. $\beta$ for Different Values '
            rf"of $\alpha$. Problem Size: $m={d['m']}, n={d['n']}$. "
            rf"$\eta={d['eta']}$",
            fontsize='x-large'
        )
        ax_it = ax_it.flatten()

        # Loop over backtracking parameters.
        for idx, alpha in enumerate(alpha_array):
            # Track iterations.
            it_count = []
            for beta in beta_array:
                # Perform gradient descent.
                result = gradient_descent(
                    x_0=d['x_0'], a=d['a'], eta=d['eta'], alpha=alpha,
                    beta=beta, it_max=d['it_max'])

                print(f"Solved (grad desc) for m={d['m']}, n={d['n']}, "
                      f"alpha={alpha:.2f}, beta={beta:.2f}")

                # Track number of iterations.
                it_count.append(len(result[1]))

            # Plot.
            ax = ax_it[idx]
            ax.text(0.08, 0.8, rf'$\mathbf{{\alpha={alpha:.2f}}}$',
                    transform=ax.transAxes, fontsize='large',
                    fontweight='bold')
            # ax.set_title(rf'$\alpha={alpha:.2f}$')
            ax.plot(beta_array, it_count, linewidth=2)
            ax.set_xlabel(r'$\beta$')
            ax.set_xticks(beta_array)
            # Label our y-axes on the left.
            if idx % 3 == 0:
                ax.set_ylabel('Number of Iterations')
            ax.grid(True)

        # Tighten the final layout.
        fig.tight_layout(h_pad=0, w_pad=0, pad=0, rect=[0, 0, 1, 0.9])
        fig.savefig(f"figs/alpha_beta_it_{d['m']}_{d['n']}.eps",
                    orientation='landscape', format='eps')
        plt.close(fig)
        print(f"Done looping over alpha and beta for m={d['m']}, n={d['n']}")


def part_b_and_c(instances):
    """Do all work for parts b and c. Use damped Newton to solve,
    plot f-p* and step length vs. iteration number.
    """
    for d in instances:
        ################################################################
        # Regular damped Newton - use full Hessian and update for
        # every iteration.
        x, obj_array, t_list = damped_newton(
            **d, diag_h=False, hessian_update=1)
        print(f"Done with regular damped Newton, m={d['m']}, n={d['n']}")

        # Plot.
        plot_results(
            obj_array=obj_array, t_list=t_list, method='Regular Damped Newton',
            param_str=d['param_str_eps'], method_file='newton_reg', m=d['m'],
            n=d['n'], p_s=d['p*'])

        ################################################################
        # Damped Newton, but evaluate and update the Hessian every
        # 3 iterations.
        x, obj_array, t_list = damped_newton(
            **d, diag_h=False, hessian_update=3)
        print(f"Done with occasional Hessian update, m={d['m']}, n={d['n']}")

        # Plot.
        plot_results(
            obj_array=obj_array, t_list=t_list,
            method='Newton Delayed Hessian Update',
            param_str=d['param_str_eps'],
            method_file='newton_delayed_h', m=d['m'], n=d['n'], p_s=d['p*'])

        ################################################################
        # Damped Newton, but only use the diagonal of the Hessian.
        x, obj_array, t_list = damped_newton(
            **d, diag_h=True, hessian_update=1)

        # Plot.
        plot_results(
            obj_array=obj_array, t_list=t_list, method='Newton Diagonal',
            param_str=d['param_str_eps'], method_file='newton_diag', m=d['m'],
            n=d['n'], p_s=d['p*'])

        print(f"Done with diagonal Hessian Newton, m={d['m']}, n={d['n']}")


def plot_results(obj_array, t_list, method, param_str, method_file, m, n, p_s):
    """Helper to plot objective value vs. iterations, step size vs.
    iterations, and optimality gap vs. iterations."""

    # Plot objective value vs. iterations.
    fig = plt.figure()
    ax = fig.gca()
    ax.semilogy(np.abs(obj_array), linewidth=2)
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel(r'$|f(\mathbf{x})|$ (Log Scale)')
    ax.set_title('Objective Value (Log Scale) vs. Iterations\nMethod: '
                 + method + param_str)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"figs/obj_vs_it_{method_file}_{m}_{n}.eps", format='eps')
    plt.close(fig)

    # Plot step size vs. iteration number.
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(t_list, linewidth=2)
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel(r'Step Size, $t$')
    ax2.set_title('Step Size vs. Iteration Number\nMethod: ' + method
                  + param_str)
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(f"figs/step_vs_it_{method_file}_{m}_{n}.eps", format='eps')
    plt.close(fig2)

    # Plot the optimality gap.
    fig3 = plt.figure()
    ax3 = fig3.gca()
    ax3.semilogy(obj_array - p_s, linewidth=2)
    ax3.set_xlabel('Iteration Number')
    ax3.set_ylabel(r'$f(\mathbf{x}) - p^*$ (Log Scale)')
    ax3.set_title('Optimality Gap vs. Iterations\nMethod: ' + method
                  + param_str)
    ax3.grid(True)
    fig3.tight_layout()
    fig3.savefig(f"figs/op_gap_vs_it_{method_file}_{m}_{n}.eps", format='eps')
    plt.close(fig3)


def _get_param_size_str(m, n, alpha, beta, eta=None, eps=None):
    """Helper for creating string of problem size and parameters for
    plot titles.
    """
    size_str = fr'Size: $m={m}, n={n}$'
    param_str = fr'Parameters: $\alpha={alpha}, \beta={beta}'

    if eta is not None:
        param_str += fr', \eta={eta}$'

    if eps is not None:
        param_str += fr', \epsilon={eps}$'

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

    # Add the x term to the diagonal.
    diag_indices = np.diag_indices(h.shape[0])
    h[diag_indices] += x_term

    # Loop to fill in the diagonal
    # TODO: This also feels like it should be vectorized...
    for k in range(len(x)):
        # Loop to compute the summation term.
        for i in range(a.shape[0]):
            h[k, k] += a_sq[i, k] / one_m_b_2[i]

    return h, one_m_b_2


def gradient_descent(x_0, a, eta, alpha, beta, it_max, *args, **kwargs):
    """Perform simple gradient descent with back-tracking line search.
    """
    # Get a copy of x_0 so we don't modify it for other project parts.
    x = x_0.copy()

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
    :param alpha: backtracking parameter. 0 < alpha < 0.5
    :param beta: backtracking parameter. 0 < beta < 1
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


def _get_hessian(x, a, diag=False):
    """Simple helper for getting the Hessian."""
    if diag:
        h, _ = diag_hessian(x, a)
        return h
    else:
        return hessian(x, a)


def damped_newton(x_0, a, eps, alpha, beta, it_max, hessian_update=1,
                  diag_h=False, *args, **kwargs):
    """Use damped Newton's method to solve."""
    # Get a copy of x_0 so we don't modify it for other project parts.
    x = x_0.copy()

    # Get initial gradient and Hessian.
    g = gradient(x, a)
    h = _get_hessian(x, a, diag_h)

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
        if (i + 1) % hessian_update == 0:
            h = _get_hessian(x, a, diag_h)
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
