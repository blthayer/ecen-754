import numpy as np


def p2():
    # Initialize b_inverse to be the identity matrix.
    b_inv = np.identity(15)

    def subtract_row(r1, r2):
        """Given 1-based indices, subtract r1 from r2"""
        b_inv[r2-1, :] = b_inv[r2-1, :] - b_inv[r1-1]
        y_0[r2-1] = y_0[r2-1] - y_0[r1-1]

    # Most of our y_0 elements are 1, so start there.
    y_0 = np.ones(15)
    # Add all the twos.
    y_0[0] = 2
    y_0[2] = 2
    y_0[6] = 2
    y_0[8] = 2
    y_0[9] = 2
    y_0[14] = 2

    # First path: e1, e2, e3
    y_1 = np.zeros_like(y_0)
    y_1[0] = 1
    y_1[1] = 1
    y_1[2] = 1

    # Pivot:
    subtract_row(2, 1)
    subtract_row(2, 3)

    # Compute lambda
    c = np.zeros_like(y_0)
    c[1] = -1
    lam = np.matmul(c, b_inv)

    # Second path: e7, e8, e9
    subtract_row(8, 7)
    subtract_row(8, 9)

    # Compute lambda
    c[7] = -1
    lam = np.matmul(c, b_inv)

    # Third path: e10, e14, e15
    subtract_row(14, 10)
    subtract_row(14, 15)

    # Compute lambda
    c[13] = -1

    # Fourth path: e1, e5, e9
    subtract_row(5, 1)
    subtract_row(5, 9)

    # Compute lambda
    c[4] = -1
    lam = np.matmul(c, b_inv)

    # Fifth path: e10, e12, e9
    subtract_row(9, 10)
    subtract_row(9, 12)

    c[8] = -1
    lam = np.matmul(c, b_inv)

    # Sixth path: e1, e5, e6, e3
    subtract_row(1, 3)
    subtract_row(1, 5)
    subtract_row(1, 6)

    c[0] = -1
    lam = np.matmul(c, b_inv)

    pass


if __name__ == '__main__':
    p2()