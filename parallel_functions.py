import sys
import io
import numpy as np
import itertools
from bposd.css import css_code

### Surpress the print statements of the qcode.test() function
# Save the current stdout so we can restore it later
original_stdout = sys.stdout
# Redirect stdout to a dummy StringIO object
sys.stdout = io.StringIO()


def get_net_encoding_rate(k, n):
    return k / (2 * n)


def process_combination(args):
    (
        ell,
        m,
        fixed_x_exponent_a,
        fixed_y_exponent_b,
        num_summands_a,
        num_summands_b,
    ) = args
    good_configs = []
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    x, y = {}, {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for j in range(m):
        y[j] = np.kron(I_ell, np.roll(I_m, j, axis=1))

    x_exponent = [i for i in range(ell) if i != fixed_x_exponent_a]
    y_exponent = [j for j in range(m) if j != fixed_y_exponent_b]

    for combo_a in itertools.combinations(y_exponent, num_summands_a - 1):
        for combo_b in itertools.combinations(x_exponent, num_summands_b - 1):
            A = x[fixed_x_exponent_a]
            for idx in combo_a:
                A += y[idx]
            A %= 2

            B = y[fixed_y_exponent_b]
            for idx in combo_b:
                B += x[idx]
            B %= 2

            # Placeholder for additional processing, like constructing and testing qcodes

            # Construct polynomial sum strings for A and B
            A_poly_sum = (
                f"x{fixed_x_exponent_a}"
                + " + "
                + " + ".join(f"y{idx}" for idx in combo_a)
            )
            B_poly_sum = (
                f"y{fixed_y_exponent_b}"
                + " + "
                + " + ".join(f"x{idx}" for idx in combo_b)
            )

            AT = np.transpose(A)
            BT = np.transpose(B)

            hx = np.hstack((A, B))
            hz = np.hstack((BT, AT))

            qcode = css_code(hx, hz)
            r = get_net_encoding_rate(qcode.K, qcode.N)

            if qcode.test() and r > 1 / 15:
                code_config = {
                    "ell": ell,
                    "m": m,
                    "n_phys_qubits": qcode.N,
                    "n_log_qubits": qcode.K,
                    "encoding_rate": r,
                    "A_poly_sum": A_poly_sum,
                    "B_poly_sum": B_poly_sum,
                }
                good_configs.append(code_config)

    return good_configs
