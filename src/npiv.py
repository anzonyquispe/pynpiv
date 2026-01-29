import numpy as np
from scipy.linalg import pinv, sqrtm
from scipy import linalg
import pandas as pd
from tqdm import tqdm
import itertools
import warnings
from util_npiv import dimbs
from prodspline import prodspline
from scipy import linalg
import math
from functools import reduce
import patsy
import matplotlib.pyplot as plt
import time 

warnings.filterwarnings("ignore")

# Re-use the B-spline basis function from previous code
def get_marginal_basis(x, degree, segments, knots_type='quantiles', x_min=None, x_max=None):
    x = np.asarray(x).ravel()
    n = len(x)
    if degree == 0:
        return np.ones((n, 1))

    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    interior_knots = None
    if segments > 0:
        if knots_type == 'quantiles':
            probs = np.linspace(0, 1, segments + 2)[1:-1]
            interior_knots = np.quantile(x, probs)
        else:  # uniform
            interior_knots = np.linspace(x_min, x_max, segments + 2)[1:-1]

    df = pd.DataFrame({'x': x})
    knots_str = 'None' if interior_knots is None else str(list(interior_knots))

    formula = (f"bs(x, df={degree + segments + 1}, degree={degree}, "
               f"include_intercept=False, "
               f"knots={knots_str})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        basis = patsy.dmatrix(formula, df, return_type='dataframe').values

    return basis

def build_basis(X, degree, segments_per_var, basis_type='additive', x_min=None, x_max=None, eval_X=None, knots='quantiles'):
    X = np.atleast_2d(X)
    n, p = X.shape
    eval_X = np.atleast_2d(eval_X if eval_X is not None else X)

    marginals = []
    for j in range(p):
        seg = segments_per_var[j] if isinstance(segments_per_var, (list, np.ndarray)) else segments_per_var
        xmin = x_min[j] if x_min is not None and len(x_min) > j else None
        xmax = x_max[j] if x_max is not None and len(x_max) > j else None
        marg = get_marginal_basis(X[:, j], degree, seg, knots, xmin, xmax)
        marginals.append(marg)

    if basis_type == 'tensor':
        basis_train = marginals[0]
        for m in marginals[1:]:
            basis_train = np.kron(basis_train, m)
        # Evaluation grid
        marginals_eval = []
        for j in range(p):
            marg_eval = get_marginal_basis(eval_X[:, j], degree, seg, knots, xmin, xmax)
            marginals_eval.append(marg_eval)
        basis_eval = marginals_eval[0]
        for m in marginals_eval[1:]:
            basis_eval = np.kron(basis_eval, m)
    else:  # additive or glp
        basis_train = np.hstack(marginals)
        basis_train = np.column_stack([np.ones(n), basis_train])
        basis_eval = np.hstack([get_marginal_basis(eval_X[:, j], degree, seg, knots, xmin, xmax) for j in range(p)])
        basis_eval = np.column_stack([np.ones(len(eval_X)), basis_eval])

    return basis_train, basis_eval

def nzd(x, tol=1e-12):
    """Replace near-zero with tol to avoid division by zero"""
    x = np.asarray(x)
    return np.where(np.abs(x) < tol, tol, x)

def npivJ(Y, X, W,
          X_grid=None,
          boot_draws_file=None,
          J_x_degree=3,
          K_w_degree=4,
          J_x_segments_set=None,
          K_w_segments_set=None,
          knots='uniform',
          basis='tensor',
          X_min=None, X_max=None, W_min=None, W_max=None,
          grid_num=50,
          boot_num=99,
          alpha=0.5,
          check_is_fullrank=False,
          progress=True):
    """
    Python translation of R's npivJ() - Adaptive NPIV via Lepski method with wild bootstrap.
    """
    Y = np.asarray(Y).ravel()
    X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    n, p_x = X.shape
    _, p_w = W.shape

    if Y.shape[0] != n:
        raise ValueError("Y, X, W must have same number of rows")
    if K_w_degree < J_x_degree:
        raise ValueError("K_w_degree must be >= J_x_degree")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")

    if check_is_fullrank:
        if np.linalg.matrix_rank(Y.reshape(-1,1)) < Y.size:
            raise ValueError("Y not full rank")
        if np.linalg.matrix_rank(X) < p_x:
            raise ValueError("X not full rank")
        if np.linalg.matrix_rank(W) < p_w:
            raise ValueError("W not full rank")

    # Default grid if not provided
    if X_grid is None:
        X_grid = np.zeros((grid_num, p_x))
        for j in range(p_x):
            col_min, col_max = X[:, j].min(), X[:, j].max()
            X_grid[:, j] = np.linspace(col_min, col_max, grid_num)

    X_grid = np.atleast_2d(X_grid)

    # Default segment sets
    if J_x_segments_set is None:
        J_x_segments_set = np.array([1, 3, 7, 15, 31, 63])  # (2^k - 1)
    if K_w_segments_set is None:
        K_w_segments_set = np.array([1, 3, 7, 15, 31, 63])

    J_x_segments_set = np.asarray(J_x_segments_set)
    K_w_segments_set = np.asarray(K_w_segments_set)

    # Generate all pairs (J1 < J2)
    pairs = [(s1, s2) for i, s1 in enumerate(J_x_segments_set) for s2 in J_x_segments_set[i+1:]]
    n_pairs = len(pairs)

    Z_sup = np.full(n_pairs, np.nan)
    Z_sup_boot = np.full((boot_num, n_pairs), np.nan)

    if progress:
        pair_pbar = tqdm(total=n_pairs, desc="Complexity determination", leave=True)

    np.random.seed(123)  # For reproducibility across bootstrap

    # Cargar bootstrap draws de R si se proporciona archivo
    if boot_draws_file is not None:
        boot_draws_R = pd.read_csv(boot_draws_file).values
    else:
        boot_draws_R = None

    for idx, (J1_seg, J2_seg) in enumerate(pairs):
        if progress:
            pair_pbar.update(1)

        # Corresponding instrument segments
        i1 = np.where(J_x_segments_set == J1_seg)[0][0]
        i2 = np.where(J_x_segments_set == J2_seg)[0][0]
        K1_seg = K_w_segments_set[i1]
        K2_seg = K_w_segments_set[i2]

        # Build instrument bases using prodspline (matches R exactly)
        if K_w_degree == 0:
            B_w_J1 = B_w_J2 = np.ones((n, 1))
        else:
            K_w_mat_J1 = np.column_stack([np.repeat(K_w_degree, p_w), np.repeat(K1_seg, p_w)])
            K_w_mat_J2 = np.column_stack([np.repeat(K_w_degree, p_w), np.repeat(K2_seg, p_w)])
            B_w_J1, _ = prodspline(x=W, K=K_w_mat_J1, knots=knots, basis=basis, x_min=W_min, x_max=W_max)
            B_w_J2, _ = prodspline(x=W, K=K_w_mat_J2, knots=knots, basis=basis, x_min=W_min, x_max=W_max)
            if basis != 'tensor':
                B_w_J1 = np.column_stack([np.ones((n, 1)), B_w_J1])
                B_w_J2 = np.column_stack([np.ones((n, 1)), B_w_J2])

        # Build X bases (on data and grid) using prodspline
        K_x_mat_J1 = np.column_stack([np.repeat(J_x_degree, p_x), np.repeat(J1_seg, p_x)])
        K_x_mat_J2 = np.column_stack([np.repeat(J_x_degree, p_x), np.repeat(J2_seg, p_x)])
        
        Psi_x_J1, _ = prodspline(x=X, K=K_x_mat_J1, knots=knots, basis=basis, x_min=X_min, x_max=X_max)
        Psi_x_J2, _ = prodspline(x=X, K=K_x_mat_J2, knots=knots, basis=basis, x_min=X_min, x_max=X_max)
        Psi_x_J1_eval, _ = prodspline(x=X, xeval=X_grid, K=K_x_mat_J1, knots=knots, basis=basis, x_min=X_min, x_max=X_max)
        Psi_x_J2_eval, _ = prodspline(x=X, xeval=X_grid, K=K_x_mat_J2, knots=knots, basis=basis, x_min=X_min, x_max=X_max)
        
        if basis != 'tensor':
            Psi_x_J1 = np.column_stack([np.ones((n, 1)), Psi_x_J1])
            Psi_x_J2 = np.column_stack([np.ones((n, 1)), Psi_x_J2])
            Psi_x_J1_eval = np.column_stack([np.ones((len(X_grid), 1)), Psi_x_J1_eval])
            Psi_x_J2_eval = np.column_stack([np.ones((len(X_grid), 1)), Psi_x_J2_eval])

        # Precompute inverses
        P_w_J1 = pinv(B_w_J1.T @ B_w_J1)
        P_w_J2 = pinv(B_w_J2.T @ B_w_J2)

        # Projection matrices
        M1 = B_w_J1 @ P_w_J1 @ B_w_J1.T
        M2 = B_w_J2 @ P_w_J2 @ B_w_J2.T

        # Reduced-form projections
        tmp1 = pinv(Psi_x_J1.T @ M1 @ Psi_x_J1) @ (Psi_x_J1.T @ M1)
        tmp2 = pinv(Psi_x_J2.T @ M2 @ Psi_x_J2) @ (Psi_x_J2.T @ M2)

        beta1 = tmp1 @ Y
        beta2 = tmp2 @ Y

        h1 = Psi_x_J1_eval @ beta1
        h2 = Psi_x_J2_eval @ beta2

        U1 = Y - Psi_x_J1 @ beta1
        U2 = Y - Psi_x_J2 @ beta2

        # Asymptotic variance components
        U1_vec = np.asarray(U1).ravel()
        A1 = tmp1.T * U1_vec[:, None]
        D1 = A1.T @ A1

        U2_vec = np.asarray(U2).ravel()
        A2 = tmp2.T * U2_vec[:, None]
        D2 = A2.T @ A2


        asy_var1 = np.sum((Psi_x_J1_eval @ D1) * Psi_x_J1_eval, axis=1)
        asy_var2 = np.sum((Psi_x_J2_eval @ D2) * Psi_x_J2_eval, axis=1)

        # Covariance term
        U1_vec = np.asarray(U1).ravel()
        U2_vec = np.asarray(U2).ravel()

        A1 = tmp1.T * U1_vec[:, None]    # n x J1
        A2 = tmp2.T * U2_vec[:, None]    # n x J2

        D1 = A1.T @ A1                   # J1 x J1
        D2 = A2.T @ A2                   # J2 x J2
        cross_term = A1.T @ A2 

        asy_cov = np.sum((Psi_x_J1_eval @ cross_term) * Psi_x_J2_eval, axis=1)

        asy_se = np.sqrt(asy_var1 + asy_var2 - 2 * asy_cov)
        t_stat = np.abs(h1 - h2) / nzd(asy_se)
        Z_sup[idx] = np.max(t_stat)

        # Wild bootstrap
        if progress and idx == 0:
            boot_pbar = tqdm(total=boot_num, desc=f"Bootstrapping pair {idx+1}/{n_pairs}", leave=False)

        for b in range(boot_num):
            if progress and idx == 0:
                boot_pbar.update(1)
            if boot_draws_R is not None:
                eps = boot_draws_R[b, :]
            else:
                eps = np.random.normal(size=n)
            num1 = Psi_x_J1_eval @ (tmp1 @ (U1 * eps))
            num2 = Psi_x_J2_eval @ (tmp2 @ (U2 * eps))
            Z_sup_boot[b, idx] = np.max(np.abs(num1 - num2) / nzd(asy_se))

        if progress and idx == 0:
            boot_pbar.close()

    if progress:
        pair_pbar.close()

    # Critical value from bootstrap
    Z_boot_max = np.max(Z_sup_boot, axis=1)
    theta_star = np.quantile(Z_boot_max, 1 - alpha)

    # Lepski's method
    num_J = len(J_x_segments_set)
    test_mat = np.zeros((num_J, num_J), dtype=bool)

    for idx, (J1_seg, J2_seg) in enumerate(pairs):
        i1 = np.where(J_x_segments_set == J1_seg)[0][0]
        i2 = np.where(J_x_segments_set == J2_seg)[0][0]
        if Z_sup[idx] <= 1.1 * theta_star:
            test_mat[i1, i2] = True

    test_vec = np.array([
        np.all(test_mat[i, i+1:]) if (i+1 < num_J) else False
        for i in range(num_J - 1)
    ] + [False])

    if np.all(~test_vec):
        J_seg = J_x_segments_set[-1]
    else:
        valid = np.where(test_vec)[0]
        J_seg = J_x_segments_set[valid[0]] if len(valid) > 0 else J_x_segments_set[0]

    # Dimension of selected basis
    def dim_basis(segments):
        per_var = J_x_degree + segments + 1
        return per_var ** p_x if basis == 'tensor' else 1 + p_x * per_var

    J_hat = dim_basis(J_seg)

    # Next-to-largest
    if len(J_x_segments_set) > 1:
        J_seg_n = np.max(np.delete(J_x_segments_set, np.argmax(J_x_segments_set)))
    else:
        J_seg_n = J_seg
    J_hat_n = dim_basis(J_seg_n)

    J_tilde = min(J_hat, J_hat_n)
    J_x_seg_final = min(J_seg, J_seg_n)
    K_w_seg = K_w_segments_set[np.where(J_x_segments_set == J_x_seg_final)[0][0]]

    return {
        'J.tilde': int(J_tilde),
        'J.hat': int(J_hat),
        'J.hat.n': int(J_hat_n),
        'J.x.seg': int(J_x_seg_final),
        'K.w.seg': int(K_w_seg),
        'theta.star': float(theta_star),
        'Z.sup': Z_sup.tolist(),
        'critical.value': float(theta_star)
    }

def is_fullrank(A, tol=1e-10):
    """
    Check if matrix A has full column rank.
    """
    A = np.asarray(A)
    return np.linalg.matrix_rank(A, tol=tol) == A.shape[1]

def sqrtm2(A, tol=1e-12):
    """
    Symmetric matrix square root of a (pseudo-)inverse / PSD matrix.
    A ≈ Q diag(sqrt(λ)) Q'.
    """
    A = np.asarray(A)
    A = 0.5 * (A + A.T)   # force symmetry
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals_clipped = np.clip(eigvals, tol, None)
    return eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T

def get_marginal_basis(x, degree, segments, knots_type='quantiles', x_min=None, x_max=None):
    """
    Generate univariate B-spline basis (interior knots = segments).
    Returns n × (degree + segments + 1) matrix.
    """
    x = np.ravel(x)
    n = len(x)

    if degree == 0:
        return np.ones((n, 1))

    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)

    interior_num = segments

    if interior_num == 0:
        interior_knots = None
    elif knots_type == 'quantiles':
        ps = np.linspace(0, 100, interior_num + 2)[1:-1]
        interior_knots = np.percentile(x, ps)
    else:  # uniform
        interior_knots = np.linspace(x_min, x_max, interior_num + 2)[1:-1]

    df = pd.DataFrame({'x': x})

    # patsy automatically handles knots=None (falls back to quantile placement if df used, but we control with knots)
    formula = f"bs(x, degree={degree}, knots={interior_knots.tolist() if interior_knots is not None else None}, include_intercept=False, lower_bound={x_min}, upper_bound={x_max})"

    basis = patsy.dmatrix(formula, df, return_type='matrix')

    return basis

def npiv_Jhat_max(X, W, J_x_degree=3, K_w_degree=4, K_w_smooth=2, knots='quantiles', basis='additive', 
                  X_min=None, W_min=None, X_max=None, W_max=None, check_is_fullrank=False, progress=True):
    """
    Python translation of the R function npiv_Jhat_max.
    Determines the maximum feasible grid resolution for nonparametric IV series estimation.
    Supports basis = 'tensor' and 'additive' (glp is treated as additive for compatibility).
    """
    X = np.atleast_2d(X)
    W = np.atleast_2d(W)

    if X.shape[0] != W.shape[0]:
        raise ValueError("X and W must have the same number of rows")

    n, p = X.shape
    _, q = W.shape

    if K_w_degree < J_x_degree:
        raise ValueError("K_w_degree must be >= J_x_degree")

    if check_is_fullrank:
        if np.linalg.matrix_rank(X) != p:
            raise ValueError("X is not of full column rank")
        if np.linalg.matrix_rank(W) != q:
            raise ValueError("W is not of full column rank")

    # Compute L.max (maximum resolution level)
    L_max = max(math.floor(math.log(n) / math.log(2 * p)), 3) if p > 0 else 3

    levels = np.arange(0, L_max + 1)
    J_x_segments_set = 2 ** levels
    K_w_segments_set = 2 ** (levels + K_w_smooth)

    test_val = np.full(len(levels), np.nan)

    pbar = tqdm(total=len(levels), desc=" Grid determination", disable=not progress)

    for i in range(len(levels)):
        pbar.update(1)

        if i <= 1 or (i > 1 and test_val[i-1] <= 10 * np.sqrt(n)):
            J_x_segments = int(J_x_segments_set[i])
            K_w_segments = int(K_w_segments_set[i])

            # === Build B_w_J (instrument basis) ===
            marginals_w = []
            for j in range(q):
                w_col = W[:, j]
                w_min = W_min[j] if W_min is not None and j < len(W_min) else None
                w_max = W_max[j] if W_max is not None and j < len(W_max) else None
                marginals_w.append(get_marginal_basis(w_col, K_w_degree, K_w_segments, knots, w_min, w_max))

            if basis == 'tensor':
                B_w_J = marginals_w[0]
                for m in marginals_w[1:]:
                    B_w_J = np.kron(B_w_J, m)
            else:  # additive or glp → same treatment
                B_w_J = np.hstack(marginals_w)
                B_w_J = np.c_[np.ones(n), B_w_J]  # cbind(1, ...)

            # === Build Psi_x_J (endogenous / reduced-form basis) ===
            marginals_x = []
            for j in range(p):
                x_col = X[:, j]
                x_min = X_min[j] if X_min is not None and j < len(X_min) else None
                x_max = X_max[j] if X_max is not None and j < len(X_max) else None
                marginals_x.append(get_marginal_basis(x_col, J_x_degree, J_x_segments, knots, x_min, x_max))

            if basis == 'tensor':
                Psi_x_J = marginals_x[0]
                for m in marginals_x[1:]:
                    Psi_x_J = np.kron(Psi_x_J, m)
            else:
                Psi_x_J = np.hstack(marginals_x)
                Psi_x_J = np.c_[np.ones(n), Psi_x_J]

            # Special case for reduced-form (perfect IV)
            if np.array_equal(X, W):
                s_hat_J = max(1.0, (0.1 * np.log(n)) ** 4)
            else:
                gram_psi = Psi_x_J.T @ Psi_x_J
                gram_b   = B_w_J.T @ B_w_J
                cross    = Psi_x_J.T @ B_w_J

                inv_sqrt_psi = sqrtm(pinv(gram_psi))
                inv_sqrt_b   = sqrtm(pinv(gram_b))

                mat = inv_sqrt_psi @ cross @ inv_sqrt_b
                s = linalg.svd(mat, compute_uv=False)
                s_hat_J = np.min(s)

            # Dimension per marginal
            dim_per_var = J_x_degree + 1 + J_x_segments

            if basis == 'tensor':
                J_x_dim = dim_per_var ** p
            else:
                J_x_dim = 1 + p * dim_per_var

            test_val[i] = J_x_dim * np.sqrt(np.log(J_x_dim)) * max((0.1 * np.log(n)) ** 4, 1.0 / s_hat_J)
        else:
            test_val[i] = test_val[i-1]

    pbar.close()

    # Find largest feasible level (largest i such that test_val[i] <= 10*sqrt(n))
    threshold = 10 * np.sqrt(n)
    feasible = np.where(test_val <= threshold)[0]
    if len(feasible) == 0:
        l_hat_max = 0
    else:
        l_hat_max = feasible.max() + 1  # number of feasible levels (1-based count)

    if l_hat_max == 0:  # safety
        l_hat_max = len(levels)

    J_x_segments_set = J_x_segments_set[:l_hat_max]
    K_w_segments_set = K_w_segments_set[:l_hat_max]

    # J.hat.max = dimension at the highest resolution
    max_segments = J_x_segments_set[-1] if len(J_x_segments_set) > 0 else 1
    dim_per_var = J_x_degree + 1 + max_segments
    if basis == 'tensor':
        J_hat_max = dim_per_var ** p
    else:
        J_hat_max = 1 + p * dim_per_var

    alpha_hat = min(0.5, np.sqrt(np.log(J_hat_max) / J_hat_max))

    return {
        'J.x.segments.set': J_x_segments_set.tolist(),
        'K.w.segments.set': K_w_segments_set.tolist(),
        'J.hat.max': int(J_hat_max),
        'alpha.hat': alpha_hat
    }

def npiv_choose_J(Y, X, W,
                  X_grid=None,
                  boot_draws_file=None,
                  J_x_degree=3, K_w_degree=4, K_w_smooth=2,
                  knots='uniform', basis='tensor',
                  X_min=None, X_max=None, W_min=None, W_max=None,
                  grid_num=50, boot_num=99,
                  npiv_Jhat_max = npiv_Jhat_max,
                  check_is_fullrank=False, progress=True):
    """
    Fully data-driven choice of tuning parameters for nonparametric IV
    via the two-step procedure in the original R package npiv.
    """

    # Step 1: Determine feasible grid and alpha
    tmp1 = npiv_Jhat_max(X, W,
                         J_x_degree=J_x_degree,
                         K_w_degree=K_w_degree,
                         K_w_smooth=K_w_smooth,
                         knots=knots,
                         basis=basis,
                         X_min=X_min, X_max=X_max,
                         W_min=W_min, W_max=W_max,
                         check_is_fullrank=check_is_fullrank,
                         progress=progress)

    # Step 2: Lepski adaptation using the grid from Step 1
    tmp2 = npivJ(Y, X, W,
                 X_grid=X_grid,
                 boot_draws_file=boot_draws_file,
                 J_x_degree=J_x_degree,
                 K_w_degree=K_w_degree,
                 J_x_segments_set=tmp1['J.x.segments.set'],
                 K_w_segments_set=tmp1['K.w.segments.set'],
                 knots=knots,
                 basis=basis,
                 X_min=X_min, X_max=X_max,
                 W_min=W_min, W_max=W_max,
                 grid_num=grid_num,
                 boot_num=boot_num,
                 alpha=tmp1['alpha.hat'],
                 check_is_fullrank=check_is_fullrank,
                 progress=progress)

    # Combine results
    result = {
        'J.hat.max': tmp1['J.hat.max'],
        'J.hat.n': tmp2['J.hat.n'],
        'J.hat': tmp2['J.hat'],
        'J.tilde': tmp2['J.tilde'],
        'J.x.seg': tmp2['J.x.seg'],
        'K.w.seg': tmp2['K.w.seg'],
        'J.x.segments.set': tmp1['J.x.segments.set'],
        'K.w.segments.set': tmp1['K.w.segments.set'],
        'theta.star': tmp2['theta.star']
    }

    return result

def is_fullrank(A, tol=1e-10):
    A = np.asarray(A)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return np.linalg.matrix_rank(A, tol=tol) == A.shape[1]

def NZD(x, eps=1e-8):
    """
    'Non-zero denominator': replace very small entries by eps
    to avoid division-by-zero in t-stats.
    """
    x = np.asarray(x, dtype=float)
    out = x.copy()
    out[np.abs(out) < eps] = eps
    return out

def quantile_type5(x, q):
    """
    R's quantile(x, probs=q, type=5, names=FALSE) for 0<q<1.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("quantile_type5: empty vector")

    h = n * q + 0.5
    if h <= 0.5:
        return float(x[0])
    if h >= n + 0.5:
        return float(x[-1])

    j = int(np.floor(h) - 1)  # 0-based index
    gamma = h - (j + 1)
    return float((1.0 - gamma) * x[j] + gamma * x[j + 1])

def npivEst(
    Y,
    X,
    W,
    X_eval=None,
    X_grid=None,
    alpha=0.05,
    basis="tensor",              # "tensor", "additive", "glp"
    boot_num=99,
    check_is_fullrank=False,
    deriv_index=1,
    deriv_order=1,
    grid_num=50,
    J_x_degree=3,
    J_x_segments=None,
    K_w_degree=4,
    K_w_segments=None,
    K_w_smooth=2,
    knots="uniform",             # "uniform", "quantiles"
    progress=True,
    ucb_h=True,
    ucb_deriv=True,
    W_max=None,
    W_min=None,
    X_min=None,
    X_max=None,
    *,
    npiv_choose_J=npiv_choose_J,          # pass your Python npiv_choose_J function
    prodspline=prodspline,             # pass your Python prodspline function
):
    """
    Python translation of npivEst R function.

    Parameters roughly match the R arguments (with underscores instead of dots).
    You must pass in `npiv_choose_J` and `prodspline` callables.
    """

    if prodspline is None:
        raise ValueError("You must pass a `prodspline` function to npivEst.")
    if npiv_choose_J is None:
        raise ValueError("You must pass an `npiv_choose_J` function to npivEst.")

    # -------------------------------------------------------------
    # Match args / basic checks
    # -------------------------------------------------------------
    if Y is None:
        raise ValueError("must provide Y")
    if X is None:
        raise ValueError("must provide X")
    if W is None:
        raise ValueError("must provide W")

    if knots not in ("uniform", "quantiles"):
        raise ValueError("knots must be 'uniform' or 'quantiles'")
    if basis not in ("tensor", "additive", "glp"):
        raise ValueError("basis must be 'tensor', 'additive', or 'glp'")

    if K_w_degree < 0:
        raise ValueError("K.w.degree must be a non-negative integer")
    if J_x_degree < 0:
        raise ValueError("J.x.degree must be a non-negative integer")
    if K_w_segments is not None and K_w_segments <= 0:
        raise ValueError("K.w.segments must be a positive integer")
    if J_x_segments is not None and J_x_segments <= 0:
        raise ValueError("J.x.segments must be a positive integer")

    Y = np.asarray(Y)
    X = np.asarray(X)
    W = np.asarray(W)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if W.ndim == 1:
        W = W.reshape(-1, 1)

    n, p = X.shape

    if check_is_fullrank:
        if not is_fullrank(Y):
            raise ValueError("Y is not of full column rank")
        if not is_fullrank(X):
            raise ValueError("X is not of full column rank")
        if not is_fullrank(W):
            raise ValueError("W is not of full column rank")

    # -------------------------------------------------------------
    # Determine if J/K chosen data-driven (npiv_choose_J) or provided
    # -------------------------------------------------------------
    check1 = False
    check2 = False

    regression_case = np.array_equal(X, W)

    if regression_case:
        # Regression case; check if J is provided
        if J_x_segments is None:
            check2 = True
    else:
        # IV case; J or K not provided
        if K_w_segments is None or J_x_segments is None:
            check1 = True

    if check1 or check2:
        data_driven = True
        # If regression, enforce K = J
        if regression_case:
            K_w_degree = J_x_degree
            K_w_smooth = 0

        # Call your Python npiv_choose_J
        test1 = npiv_choose_J(
            Y=Y,
            X=X,
            W=W,
            X_grid=X_grid,
            J_x_degree=J_x_degree,
            K_w_degree=K_w_degree,
            K_w_smooth=K_w_smooth,
            knots=knots,
            basis=basis,
            X_min=X_min,
            X_max=X_max,
            W_min=W_min,
            W_max=W_max,
            grid_num=grid_num,
            boot_num=boot_num,
            check_is_fullrank=check_is_fullrank,
            progress=progress,
        )

        # Support both R-style and Python-style keys
        def _get(d, *names):
            for nm in names:
                if nm in d:
                    return d[nm]
            raise KeyError(f"None of {names} found in npiv_choose_J output")

        K_w_segments = _get(test1, "K_w_seg", "K.w.seg")
        J_x_segments = _get(test1, "J_x_seg", "J.x.seg")
        J_x_segments_set = _get(test1, "J_x_segments_set", "J.x.segments.set")
        K_w_segments_set = _get(test1, "K_w_segments_set", "K.w.segments.set")
        J_tilde = _get(test1, "J_tilde", "J.tilde")
        theta_star = _get(test1, "theta_star", "theta.star")

        J_x_segments_set = np.asarray(J_x_segments_set, dtype=int)
        K_w_segments_set = np.asarray(K_w_segments_set, dtype=int)
    else:
        data_driven = False
        if regression_case:
            K_w_degree = J_x_degree
            K_w_segments = J_x_segments
            K_w_smooth = 0

    if K_w_degree + K_w_segments < J_x_degree + J_x_segments:
        raise ValueError("K.w.degree+K.w.segments must be >= J.x.degree+J.x.segments")

    # -------------------------------------------------------------
    # Basis for W
    # -------------------------------------------------------------
    if K_w_degree == 0:
        B_w = np.ones((n, 1))
    else:
        K_w_mat = np.column_stack([
            np.repeat(K_w_degree, W.shape[1]),
            np.repeat(K_w_segments, W.shape[1]),
        ])
        # FIX: desempaquetar tupla
        B_w, _ = prodspline(
            x=W,
            K=K_w_mat,
            knots=knots,
            basis=basis,
            x_min=W_min,
            x_max=W_max,
        )
        if basis != "tensor":
            B_w = np.column_stack([np.ones((n, 1)), B_w])

    # -------------------------------------------------------------
    # Basis for X and derivative, plus eval grid if provided
    # -------------------------------------------------------------
    if J_x_degree == 0:
        Psi_x = Psi_x_eval = np.ones((n, 1))
        Psi_x_deriv = Psi_x_deriv_eval = np.zeros((n, 1))
    else:
        K_x_mat = np.column_stack([
            np.repeat(J_x_degree, X.shape[1]),
            np.repeat(J_x_segments, X.shape[1]),
        ])

        # FIX: desempaquetar tupla
        Psi_x, _ = prodspline(
            x=X,
            K=K_x_mat,
            knots=knots,
            basis=basis,
            x_min=X_min,
            x_max=X_max,
        )
        Psi_x_eval = Psi_x.copy()

        # FIX: desempaquetar tupla
        Psi_x_deriv, _ = prodspline(
            x=X,
            K=K_x_mat,
            knots=knots,
            basis=basis,
            deriv_index=deriv_index,
            deriv_order=deriv_order,
            x_min=X_min,
            x_max=X_max,
        )
        Psi_x_deriv_eval = Psi_x_deriv.copy()

        # if X_eval is provided, evaluate basis at X_eval
        if X_eval is not None:
            X_eval = np.asarray(X_eval)
            if X_eval.ndim == 1:
                X_eval = X_eval.reshape(-1, 1)

            # FIX: desempaquetar tupla
            Psi_x_eval, _ = prodspline(
                x=X,
                xeval=X_eval,
                K=K_x_mat,
                knots=knots,
                basis=basis,
                x_min=X_min,
                x_max=X_max,
            )
            # FIX: desempaquetar tupla
            Psi_x_deriv_eval, _ = prodspline(
                x=X,
                xeval=X_eval,
                K=K_x_mat,
                knots=knots,
                basis=basis,
                deriv_index=deriv_index,
                deriv_order=deriv_order,
                x_min=X_min,
                x_max=X_max,
            )

        if basis != "tensor":
            # add constant term for function, zero for derivative
            Psi_x = np.column_stack([np.ones((Psi_x.shape[0], 1)), Psi_x])
            Psi_x_eval = np.column_stack([np.ones((Psi_x_eval.shape[0], 1)), Psi_x_eval])

            Psi_x_deriv = np.column_stack([np.zeros((Psi_x_deriv.shape[0], 1)), Psi_x_deriv])
            Psi_x_deriv_eval = np.column_stack([np.zeros((Psi_x_deriv_eval.shape[0], 1)), Psi_x_deriv_eval])

    # -------------------------------------------------------------
    # Compute beta via generalized inverse (NPIV)
    # -------------------------------------------------------------
    # Psi.xTB.wB.wTB.w.invB.w <- t(Psi.x)%*%B.w%*%ginv(t(B.w)%*%B.w)%*%t(B.w)
    Psi = Psi_x
    B = B_w

    BtB_inv = np.linalg.pinv(B.T @ B)
    PsiT_B = Psi.T @ B
    A = PsiT_B @ BtB_inv @ B.T          # J x n
    M = A @ Psi                         # J x J
    M_inv = np.linalg.pinv(M)
    tmp = M_inv @ A                     # J x n

    beta = tmp @ Y                      # J x 1

    # -------------------------------------------------------------
    # IV function and derivative at eval points
    # -------------------------------------------------------------
    h = Psi_x_eval @ beta               # n_eval x 1
    h_deriv = Psi_x_deriv_eval @ beta   # n_eval x 1

    # -------------------------------------------------------------
    # Asymptotic standard errors
    # -------------------------------------------------------------
    U_hat = Y - (Psi @ beta)            # n x 1

    # D.inv.rho.D.inv <- t(t(tmp) * as.numeric(U.hat))%*%(t(tmp) * as.numeric(U.hat))
    U_vec = U_hat.ravel()               # n,
    tmp_T = tmp.T                       # n x J
    A_weighted = tmp_T * U_vec[:, None] # n x J
    D_inv_rho_D_inv = A_weighted.T @ A_weighted  # J x J

    # asy.se <- sqrt(abs(rowSums((Psi.x.eval%*%D.inv.rho.D.inv)*Psi.x.eval)))
    M_eval = Psi_x_eval @ D_inv_rho_D_inv   # n_eval x J
    var_h = np.sum(M_eval * Psi_x_eval, axis=1)
    asy_se = np.sqrt(np.abs(var_h))

    M_eval_deriv = Psi_x_deriv_eval @ D_inv_rho_D_inv
    var_h_deriv = np.sum(M_eval_deriv * Psi_x_deriv_eval, axis=1)
    asy_se_deriv = np.sqrt(np.abs(var_h_deriv))

    # -------------------------------------------------------------
    # Uniform confidence bands via bootstrap, if requested
    # -------------------------------------------------------------
    h_lower = h_upper = cv = None
    h_lower_deriv = h_upper_deriv = cv_deriv = None

    if ucb_h or ucb_deriv:

        if data_driven:
            # ---------------------------
            # Chen, Christensen, Kankanala (2024) UCB
            # ---------------------------
            J_x_segments_set = np.asarray(J_x_segments_set, dtype=int)
            K_w_segments_set = np.asarray(K_w_segments_set, dtype=int)

            if J_x_segments_set.size > 2:
                limit = max(J_x_segments, np.max(J_x_segments_set[:-2]))
                J_x_segments_boot = J_x_segments_set[J_x_segments_set <= limit]
            else:
                J_x_segments_boot = J_x_segments_set.copy()

            L_boot = len(J_x_segments_boot)

            if ucb_h:
                Z_sup_boot = np.full((boot_num, L_boot), np.nan)
            else:
                Z_sup_boot = None
            if ucb_deriv:
                Z_sup_boot_deriv = np.full((boot_num, L_boot), np.nan)
            else:
                Z_sup_boot_deriv = None

            # Pre-generate bootstrap multipliers so they are *identical across J*
            rng = np.random.default_rng()
            boot_draws_mat = rng.standard_normal(size=(boot_num, n))

            # Optional progress
            if progress:
                try:
                    from tqdm import trange
                    outer_iter = trange(L_boot, desc="boot over J")
                except ImportError:
                    outer_iter = range(L_boot)
            else:
                outer_iter = range(L_boot)

            for ii in outer_iter:
                J_seg_i = int(J_x_segments_set[ii])
                K_seg_i = int(K_w_segments_set[ii])

                # Basis for W at this J
                if K_w_degree == 0:
                    B_w_J = np.ones((n, 1))
                else:
                    K_w_mat_i = np.column_stack([
                        np.repeat(K_w_degree, W.shape[1]),
                        np.repeat(K_seg_i, W.shape[1]),
                    ])
                    # FIX: desempaquetar tupla
                    B_w_J, _ = prodspline(
                        x=W,
                        K=K_w_mat_i,
                        knots=knots,
                        basis=basis,
                        x_min=W_min,
                        x_max=W_max,
                    )
                    if basis != "tensor":
                        B_w_J = np.column_stack([np.ones((n, 1)), B_w_J])

                # Basis for X at this J
                if J_x_degree == 0:
                    Psi_x_J = Psi_x_J_eval = np.ones((n, 1))
                    if ucb_deriv:
                        Psi_x_J_deriv = Psi_x_J_deriv_eval = np.zeros((n, 1))
                else:
                    K_x_mat_i = np.column_stack([
                        np.repeat(J_x_degree, X.shape[1]),
                        np.repeat(J_seg_i, X.shape[1]),
                    ])
                    # FIX: desempaquetar tupla
                    Psi_x_J, _ = prodspline(
                        x=X,
                        K=K_x_mat_i,
                        knots=knots,
                        basis=basis,
                        x_min=X_min,
                        x_max=X_max,
                    )
                    Psi_x_J_eval = Psi_x_J.copy()

                    if ucb_deriv:
                        # FIX: desempaquetar tupla
                        Psi_x_J_deriv, _ = prodspline(
                            x=X,
                            K=K_x_mat_i,
                            knots=knots,
                            basis=basis,
                            deriv_index=deriv_index,
                            deriv_order=deriv_order,
                            x_min=X_min,
                            x_max=X_max,
                        )
                        Psi_x_J_deriv_eval = Psi_x_J_deriv.copy()

                    if X_eval is not None:
                        # FIX: desempaquetar tupla
                        Psi_x_J_eval, _ = prodspline(
                            x=X,
                            xeval=X_eval,
                            K=K_x_mat_i,
                            knots=knots,
                            basis=basis,
                            x_min=X_min,
                            x_max=X_max,
                        )
                        if ucb_deriv:
                            # FIX: desempaquetar tupla
                            Psi_x_J_deriv_eval, _ = prodspline(
                                x=X,
                                xeval=X_eval,
                                K=K_x_mat_i,
                                knots=knots,
                                basis=basis,
                                deriv_index=deriv_index,
                                deriv_order=deriv_order,
                                x_min=X_min,
                                x_max=X_max,
                            )

                    if basis != "tensor":
                        Psi_x_J = np.column_stack([np.ones((Psi_x_J.shape[0], 1)), Psi_x_J])
                        Psi_x_J_eval = np.column_stack([np.ones((Psi_x_J_eval.shape[0], 1)), Psi_x_J_eval])

                        if ucb_deriv:
                            Psi_x_J_deriv = np.column_stack([np.zeros((Psi_x_J_deriv.shape[0], 1)),
                                                             Psi_x_J_deriv])
                            Psi_x_J_deriv_eval = np.column_stack([np.zeros((Psi_x_J_deriv_eval.shape[0], 1)),
                                                                  Psi_x_J_deriv_eval])

                # Recompute beta, residuals etc for this J
                BtB_J_inv = np.linalg.pinv(B_w_J.T @ B_w_J)
                PsiT_B_J = Psi_x_J.T @ B_w_J
                A_J = PsiT_B_J @ BtB_J_inv @ B_w_J.T
                M_J = A_J @ Psi_x_J
                M_J_inv = np.linalg.pinv(M_J)
                tmp_J = M_J_inv @ A_J   # J x n

                beta_J = tmp_J @ Y
                U_J = Y - (Psi_x_J @ beta_J)
                U_J_vec = U_J.ravel()

                # Asymptotic variances
                tmp_J_T = tmp_J.T                    # n x J
                A_weighted_J = tmp_J_T * U_J_vec[:, None]
                D_J_inv_rho_D_J_inv = A_weighted_J.T @ A_weighted_J  # J x J

                if ucb_h:
                    M_eval_J = Psi_x_J_eval @ D_J_inv_rho_D_J_inv
                    var_J = np.sum(M_eval_J * Psi_x_J_eval, axis=1)
                    asy_se_J = np.sqrt(np.abs(var_J))
                if ucb_deriv:
                    M_eval_J_deriv = Psi_x_J_deriv_eval @ D_J_inv_rho_D_J_inv
                    var_J_deriv = np.sum(M_eval_J_deriv * Psi_x_J_deriv_eval, axis=1)
                    asy_se_J_deriv = np.sqrt(np.abs(var_J_deriv))

                # Bootstrap: sup t-stat across eval points
                for b in range(boot_num):
                    boot_draws = boot_draws_mat[b]  # n,
                    z = U_J_vec * boot_draws       # n,
                    z_col = z.reshape(-1, 1)       # n x 1

                    if ucb_h:
                        vals = (Psi_x_J_eval @ (tmp_J @ z_col)).ravel()
                        t_stats = vals / NZD(asy_se_J)
                        Z_sup_boot[b, ii] = np.max(np.abs(t_stats))

                    if ucb_deriv:
                        vals_d = (Psi_x_J_deriv_eval @ (tmp_J @ z_col)).ravel()
                        t_stats_d = vals_d / NZD(asy_se_J_deriv)
                        Z_sup_boot_deriv[b, ii] = np.max(np.abs(t_stats_d))

            # Max over J for each bootstrap draw, then quantiles
            if ucb_h:
                Z_boot = np.max(Z_sup_boot, axis=1)   # boot_num
                z_star = quantile_type5(Z_boot, 1 - alpha)
                cv = z_star + max(0.0, np.log(np.log(J_tilde))) * theta_star

            if ucb_deriv:
                Z_boot_deriv = np.max(Z_sup_boot_deriv, axis=1)
                z_star_deriv = quantile_type5(Z_boot_deriv, 1 - alpha)
                cv_deriv = z_star_deriv + max(0.0, np.log(np.log(J_tilde))) * theta_star

            # Restore data-driven chosen segments (overwritten in loop)
            J_x_segments = _get(test1, "J_x_seg", "J.x.seg")
            K_w_segments = _get(test1, "K_w_seg", "K.w.seg")

        else:
            # ---------------------------
            # Chen & Christensen (2018) UCB (fixed basis dimension)
            # ---------------------------
            if ucb_h:
                Z_sup_boot = np.zeros(boot_num)
            else:
                Z_sup_boot = None
            if ucb_deriv:
                Z_sup_boot_deriv = np.zeros(boot_num)
            else:
                Z_sup_boot_deriv = None

            rng = np.random.default_rng()
            boot_draws_mat = rng.standard_normal(size=(boot_num, n))
            U_vec = U_hat.ravel()

            for b in range(boot_num):
                z = U_vec * boot_draws_mat[b]     # n,
                z_col = z.reshape(-1, 1)          # n x 1

                if ucb_h:
                    vals = (Psi_x_eval @ (tmp @ z_col)).ravel()
                    t_stats = vals / NZD(asy_se)
                    Z_sup_boot[b] = np.max(np.abs(t_stats))

                if ucb_deriv:
                    vals_d = (Psi_x_deriv_eval @ (tmp @ z_col)).ravel()
                    t_stats_d = vals_d / NZD(asy_se_deriv)
                    Z_sup_boot_deriv[b] = np.max(np.abs(t_stats_d))

            if ucb_h:
                cv = quantile_type5(Z_sup_boot, 1 - alpha)
            if ucb_deriv:
                cv_deriv = quantile_type5(Z_sup_boot_deriv, 1 - alpha)

        # Compute bands
        if ucb_h:
            h_lower = h.ravel() - cv * asy_se
            h_upper = h.ravel() + cv * asy_se
        if ucb_deriv:
            h_lower_deriv = h_deriv.ravel() - cv_deriv * asy_se_deriv
            h_upper_deriv = h_deriv.ravel() + cv_deriv * asy_se_deriv

    # -------------------------------------------------------------
    # Prepare return dict
    # -------------------------------------------------------------
    if X_eval is not None:
        nevalobs = X_eval.shape[0]
        trainiseval = False
        X_eval_return = X_eval
    else:
        nevalobs = None
        trainiseval = True
        X_eval_return = X

    result = {
        "J_x_degree": J_x_degree,
        "J_x_segments": J_x_segments,
        "K_w_degree": K_w_degree,
        "K_w_segments": K_w_segments,
        "asy_se": asy_se,
        "beta": beta,
        "cv_deriv": cv_deriv,
        "cv": cv,
        "deriv_asy_se": asy_se_deriv,
        "deriv_index": deriv_index,
        "deriv_order": deriv_order,
        "deriv": h_deriv.ravel(),
        "h_lower_deriv": h_lower_deriv,
        "h_lower": h_lower,
        "h_upper_deriv": h_upper_deriv,
        "h_upper": h_upper,
        "h": h.ravel(),
        "nevalobs": nevalobs,
        "nobs": n,
        "ndim": p,
        "residuals": (Y - Psi @ beta).ravel(),
        "trainiseval": trainiseval,
        "Y": Y,
        "X": X,
        "X_eval": X_eval_return,
        "W": W,
    }

    return result

def npiv(
    Y,
    X,
    W,
    X_eval=None,
    X_grid=None,
    alpha=0.05,
    basis="tensor",          # "tensor", "additive", "glp"
    boot_num=99,
    check_is_fullrank=False,
    deriv_index=1,
    deriv_order=1,
    grid_num=50,
    J_x_degree=3,
    J_x_segments=None,
    K_w_degree=4,
    K_w_segments=None,
    K_w_smooth=2,
    knots="uniform",         # "uniform", "quantiles"
    progress=True,
    ucb_h=True,
    ucb_deriv=True,
    W_max=None,
    W_min=None,
    X_min=None,
    X_max=None,
    **kwargs,                # extra args passed to npivEst (e.g. prodspline, npiv_choose_J)
):
    """
    Python translation of the R wrapper npiv().

    This is a thin wrapper around npivEst that:
    - measures runtime,
    - adds some metadata fields (alpha, basis, etc.),
    - tags the result as an 'npiv' object via result['class'].
    """

    t0 = time.time()

    est = npivEst(
        Y=Y,
        X=X,
        W=W,
        X_eval=X_eval,
        X_grid=X_grid,
        alpha=alpha,
        basis=basis,
        boot_num=boot_num,
        check_is_fullrank=check_is_fullrank,
        deriv_index=deriv_index,
        deriv_order=deriv_order,
        grid_num=grid_num,
        J_x_degree=J_x_degree,
        J_x_segments=J_x_segments,
        K_w_degree=K_w_degree,
        K_w_segments=K_w_segments,
        K_w_smooth=K_w_smooth,
        knots=knots,
        progress=progress,
        ucb_h=ucb_h,
        ucb_deriv=ucb_deriv,
        W_max=W_max,
        W_min=W_min,
        X_min=X_min,
        X_max=X_max,
        **kwargs,
    )

    t1 = time.time()

    # Add metadata (mimic R code)
    # "call" in R is the matched call; here we store a simple dict of main args
    est["call"] = {
        "alpha": alpha,
        "basis": basis,
        "boot_num": boot_num,
        "grid_num": grid_num,
        "knots": knots,
        "J_x_degree": J_x_degree,
        "J_x_segments": J_x_segments,
        "K_w_degree": K_w_degree,
        "K_w_segments": K_w_segments,
        "K_w_smooth": K_w_smooth,
    }

    est["ptm"] = float(t1 - t0)              # elapsed time in seconds
    est["alpha"] = alpha
    est["basis"] = basis
    est["boot_num"] = boot_num
    est["check_is_fullrank"] = check_is_fullrank
    est["grid_num"] = grid_num
    est["knots"] = knots
    est["progress"] = progress
    est["W_max"] = W_max
    est["W_min"] = W_min
    est["X_min"] = X_min
    est["X_max"] = X_max

    # Tag as "npiv" object (R: class(est) <- "npiv")
    est["class"] = "npiv"

    return est

def plot_npiv(x, kind="func", showdata=True, ax=None, **plot_kwargs):
    """
    Python translation of the R method plot.npiv.

    Parameters
    ----------
    x : dict
        Result from npivEst/npiv, expected keys:
        - "X_eval", "h", "h_lower", "h_upper"
        - "deriv", "h_lower_deriv", "h_upper_deriv"
        - "X", "Y"    (for showdata=True)
    kind : {"func", "deriv"}
        "func"  -> plot function h(x) with uniform confidence band
        "deriv" -> plot derivative h'(x) with uniform confidence band
    showdata : bool, default False
        If True and kind == "func", overlay original (X, Y) data points.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, uses current axes (plt.gca()).
    **plot_kwargs :
        Extra keyword arguments passed to the main line plot (estimate).
    """

    if kind not in ("func", "deriv"):
        raise ValueError("kind must be either 'func' or 'deriv'")

    # Extract evaluation grid
    X_eval = np.asarray(x["X_eval"])

    # Check univariate
    if X_eval.ndim > 1 and X_eval.shape[1] > 1:
        raise ValueError("plot_npiv only works for univariate X")

    # Flatten to 1D
    X_eval_1d = X_eval.ravel()
    order = np.argsort(X_eval_1d)

    if ax is None:
        ax = plt.gca()

    # Set x-limits based on eval grid
    ax.set_xlim(X_eval_1d.min(), X_eval_1d.max())

    if kind == "func":
        h = np.asarray(x["h"]).ravel()
        h_lower = np.asarray(x["h_lower"]) if x["h_lower"] is not None else None
        h_upper = np.asarray(x["h_upper"]) if x["h_upper"] is not None else None

        # Compute y-limits if bands present, else just from h
        vals = [h]
        if h_lower is not None:
            vals.append(h_lower)
        if h_upper is not None:
            vals.append(h_upper)
        ylim = (np.min(np.concatenate(vals)), np.max(np.concatenate(vals)))

        # Main function estimate
        ax.plot(
            X_eval_1d[order],
            h[order],
            color="blue",
            linestyle="-",
            label="Estimate",
            **plot_kwargs,
        )
        ax.set_ylim(ylim)
        ax.set_xlabel("X")
        ax.set_ylabel("Function")

        # Show data (X, Y) if requested
        if showdata:
            X_data = np.asarray(x["X"])
            Y_data = np.asarray(x["Y"])

            if X_data.ndim > 1:
                X_data = X_data[:, 0]
            ax.scatter(X_data, Y_data, s=5, color="lightgrey", alpha=0.7)

        # Confidence bands
        if h_lower is not None and h_upper is not None:
            ax.plot(
                X_eval_1d[order],
                h_lower[order],
                color="red",
                linestyle="--",
                label="Confidence Band",
            )
            ax.plot(
                X_eval_1d[order],
                h_upper[order],
                color="red",
                linestyle="--",
            )

        # Legend
        ax.legend(loc="upper right", frameon=False)

    else:  # kind == "deriv"
        deriv = np.asarray(x["deriv"]).ravel()
        h_lower_d = (
            np.asarray(x["h_lower_deriv"]) if x["h_lower_deriv"] is not None else None
        )
        h_upper_d = (
            np.asarray(x["h_upper_deriv"]) if x["h_upper_deriv"] is not None else None
        )

        vals = [deriv]
        if h_lower_d is not None:
            vals.append(h_lower_d)
        if h_upper_d is not None:
            vals.append(h_upper_d)
        ylim = (np.min(np.concatenate(vals)), np.max(np.concatenate(vals)))

        # Derivative estimate
        ax.plot(
            X_eval_1d[order],
            deriv[order],
            color="blue",
            linestyle="-",
            label="Estimate",
            **plot_kwargs,
        )
        ax.set_ylim(ylim)
        ax.set_xlabel("X")
        ax.set_ylabel("Derivative")

        # Confidence bands for derivative
        if h_lower_d is not None and h_upper_d is not None:
            ax.plot(
                X_eval_1d[order],
                h_lower_d[order],
                color="red",
                linestyle="--",
                label="Confidence Band",
            )
            ax.plot(
                X_eval_1d[order],
                h_upper_d[order],
                color="red",
                linestyle="--",
            )

        ax.legend(loc="upper right", frameon=False)

    return ax