import numpy as np
from scipy.stats import t


def pval_descriptives(x: float, freedom: int):
    pval = 1 - t.cdf(np.abs(x), freedom)

    if pval < 0.01:
        return ("***", pval)
    if pval < 0.05:
        return ("**", pval)
    if pval < 0.1:
        return ("*", pval)

    return ("", pval)


def pretty_print_pval(x: float, freedom: int, precision: int, parens: bool = True):
    pval_res = pval_descriptives(x, freedom)

    if parens:
        if pval_res[0]:
            return f"({x:.{precision}f}) ({pval_res[0]})"
        else:
            return f"({x:.{precision}f})"

    else:
        if pval_res[0]:
            return f"{x:.{precision}f} ({pval_res[0]})"
        else:
            return f"{x:.{precision}f}"
