"""
NPIV Engel Curve Estimation - Test Script
Replicates R's npiv::npiv() using 2SLS (Two-Stage Least Squares)

This script uses the existing modules:
- util_npiv.py
- gsl_bspline.py
- mgcv_tensor.py
- glp_model_matrix.py
- prodspline.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import pinv

# Import from existing modules
from prodspline import prodspline
from gsl_bspline import predict_gsl_bspline


def npiv_estimate(Y, X, W, 
                  J_x_degree=3, J_x_segments=1,
                  K_w_degree=4, K_w_segments=4,
                  knots="uniform",  # R's default is "uniform", not "quantiles"
                  basis="tensor",
                  X_eval=None):
    """
    Nonparametric IV estimation using 2SLS with B-spline sieves.
    Replicates R's npiv::npiv() function.
    """
    Y = np.asarray(Y).ravel()
    X = np.asarray(X).ravel()
    W = np.asarray(W).ravel()
    n = len(Y)
    
    if X_eval is None:
        X_eval = X.copy()
        trainiseval = True
    else:
        X_eval = np.asarray(X_eval).ravel()
        trainiseval = False
    
    # Step 1: Construct B-spline bases using prodspline
    K_x = np.array([[J_x_degree, J_x_segments]])
    K_w = np.array([[K_w_degree, K_w_segments]])
    
    # Basis for X (endogenous regressor) - Psi
    Psi_x, _ = prodspline(
        X.reshape(-1, 1), 
        K=K_x, 
        xeval=X.reshape(-1, 1), 
        knots=knots, 
        basis=basis
    )
    
    # Basis for X at evaluation points
    Psi_x_eval, _ = prodspline(
        X.reshape(-1, 1), 
        K=K_x, 
        xeval=X_eval.reshape(-1, 1), 
        knots=knots, 
        basis=basis
    )
    
    # Derivative basis at evaluation points
    Psi_x_deriv_eval, _ = prodspline(
        X.reshape(-1, 1), 
        K=K_x, 
        xeval=X_eval.reshape(-1, 1), 
        knots=knots, 
        basis=basis,
        deriv_index=1,
        deriv_order=1
    )
    
    # Basis for W (instrument) - B
    B_w, _ = prodspline(
        W.reshape(-1, 1), 
        K=K_w, 
        xeval=W.reshape(-1, 1), 
        knots=knots, 
        basis=basis
    )
    
    # Step 2: 2SLS Estimation (NOT OLS!)
    is_regression = np.allclose(X, W)
    
    if is_regression:
        # Simple regression: OLS
        PsiTPsi = Psi_x.T @ Psi_x
        PsiTY = Psi_x.T @ Y
        beta = pinv(PsiTPsi) @ PsiTY
    else:
        # IV estimation: 2SLS (formula from R's npiv.R lines 408-410)
        # beta = (Psi'B(B'B)^{-1}B'Psi)^{-1} Psi'B(B'B)^{-1}B'Y
        BtB = B_w.T @ B_w
        BtB_inv = pinv(BtB)
        BtPsi = B_w.T @ Psi_x
        BtY = B_w.T @ Y
        
        PsiBBinvBPsi = Psi_x.T @ B_w @ BtB_inv @ BtPsi
        PsiBBinvBY = Psi_x.T @ B_w @ BtB_inv @ BtY
        
        beta = pinv(PsiBBinvBPsi) @ PsiBBinvBY
    
    # Step 3: Fitted values and residuals
    h_train = Psi_x @ beta
    residuals = Y - h_train
    
    h_eval = Psi_x_eval @ beta
    deriv_eval = Psi_x_deriv_eval @ beta
    
    # Step 4: Standard errors (heteroskedasticity-robust, like R)
    # R uses: D.inv.rho.D.inv <- t(t(tmp) * U.hat) %*% (t(tmp) * U.hat)
    # Compute tmp matrix (projection matrix for 2SLS)
    if is_regression:
        tmp = pinv(Psi_x.T @ Psi_x) @ Psi_x.T
    else:
        BtB_inv = pinv(B_w.T @ B_w)
        PsiBBinvBPsi = Psi_x.T @ B_w @ BtB_inv @ B_w.T @ Psi_x
        tmp = pinv(PsiBBinvBPsi) @ Psi_x.T @ B_w @ BtB_inv @ B_w.T
    
    # Heteroskedasticity-robust variance (sandwich estimator)
    # R: D.inv.rho.D.inv = t(t(tmp) * U.hat) %*% (t(tmp) * U.hat)
    # tmp has shape (k, n), tmp.T has shape (n, k)
    # t(tmp) * U.hat multiplies each row of tmp.T by corresponding element of U.hat
    tmp_T_times_U = tmp.T * residuals[:, np.newaxis]  # (n, k)
    D_inv_rho_D_inv = tmp_T_times_U.T @ tmp_T_times_U  # (k, k)
    
    # asy.se = sqrt(rowSums((Psi.x.eval %*% D.inv.rho.D.inv) * Psi.x.eval))
    var_h = np.sum((Psi_x_eval @ D_inv_rho_D_inv) * Psi_x_eval, axis=1)
    se_h = np.sqrt(np.abs(var_h))
    
    var_deriv = np.sum((Psi_x_deriv_eval @ D_inv_rho_D_inv) * Psi_x_deriv_eval, axis=1)
    se_deriv = np.sqrt(np.abs(var_deriv))
    
    # Also store sigma2 for reference
    sigma2 = np.sum(residuals**2) / (n - len(beta))
    
    return {
        'h': h_eval,
        'se': se_h,
        'deriv': deriv_eval,
        'deriv_se': se_deriv,
        'beta': beta,
        'Psi_x': Psi_x,
        'Psi_x_eval': Psi_x_eval,
        'Psi_x_deriv_eval': Psi_x_deriv_eval,
        'B_w': B_w,
        'residuals': residuals,
        'sigma2': sigma2,
        'tmp': tmp,  # projection matrix for UCB
        'D_inv_rho_D_inv': D_inv_rho_D_inv,  # robust variance matrix
        'Y': Y,
        'X': X,
        'W': W,
        'X_eval': X_eval,
        'is_regression': is_regression,
        'trainiseval': trainiseval,
        'J_x_degree': J_x_degree,
        'J_x_segments': J_x_segments,
        'K_w_degree': K_w_degree,
        'K_w_segments': K_w_segments,
        'knots': knots,
        'basis': basis
    }


def npiv_ucb(npiv_result, boot_num=99, alpha=0.05, seed=None, progress=True, 
             cv_fixed=None, cv_deriv_fixed=None):
    """
    Compute Uniform Confidence Bands using Chen and Christensen (2018) method.
    This replicates EXACTLY how R's npiv computes confidence bands.
    
    Parameters
    ----------
    cv_fixed, cv_deriv_fixed : float, optional
        If provided, use these fixed critical values instead of bootstrapping.
        Useful for exact replication of R results.
    """
    
    # Extract from npiv_result
    Y = npiv_result['Y']
    X = npiv_result['X']
    X_eval = npiv_result['X_eval']
    Psi_x = npiv_result['Psi_x']
    Psi_x_eval = npiv_result['Psi_x_eval']
    Psi_x_deriv_eval = npiv_result['Psi_x_deriv_eval']
    B_w = npiv_result['B_w']
    residuals = npiv_result['residuals']
    h = npiv_result['h']
    deriv = npiv_result['deriv']
    asy_se = npiv_result['se']
    asy_se_deriv = npiv_result['deriv_se']
    tmp = npiv_result['tmp']  # Use tmp from npiv_estimate
    
    n = len(Y)
    
    # NZD function (avoid division by zero, like R)
    def NZD(a):
        eps = np.finfo(float).eps
        return np.where(a < 0, np.minimum(-eps, a), np.maximum(eps, a))
    
    # Bootstrap the sup t-stat
    if seed is not None:
        np.random.seed(seed)
    
    Z_sup_boot = np.zeros(boot_num)
    Z_sup_boot_deriv = np.zeros(boot_num)
    
    if progress:
        print(f"UCB Bootstrap ({boot_num} replications)...", end=" ", flush=True)
    
    for b in range(boot_num):
        # Draw from standard normal (like R's rnorm)
        boot_draws = np.random.randn(n)
        
        # Compute sup t-stat for h
        numerator_h = Psi_x_eval @ tmp @ (residuals * boot_draws)
        Z_sup_boot[b] = np.max(np.abs(numerator_h / NZD(asy_se)))
        
        # Compute sup t-stat for derivative
        numerator_deriv = Psi_x_deriv_eval @ tmp @ (residuals * boot_draws)
        Z_sup_boot_deriv[b] = np.max(np.abs(numerator_deriv / NZD(asy_se_deriv)))
        
        if progress and (b + 1) % 20 == 0:
            print(f"{b+1}", end=" ", flush=True)
    
    if progress:
        print("Done!")
    
    # Compute critical values (R uses type=5 quantile)
    if cv_fixed is not None:
        cv = cv_fixed
    else:
        cv = np.percentile(Z_sup_boot, 100 * (1 - alpha))
    
    if cv_deriv_fixed is not None:
        cv_deriv = cv_deriv_fixed
    else:
        cv_deriv = np.percentile(Z_sup_boot_deriv, 100 * (1 - alpha))
    
    # Compute UCBs: h ± cv * asy_se
    h_lower = h - cv * asy_se
    h_upper = h + cv * asy_se
    
    deriv_lower = deriv - cv_deriv * asy_se_deriv
    deriv_upper = deriv + cv_deriv * asy_se_deriv
    
    return {
        'h_lower': h_lower,
        'h_upper': h_upper,
        'deriv_lower': deriv_lower,
        'deriv_upper': deriv_upper,
        'cv': cv,
        'cv_deriv': cv_deriv
    }


def plot_npiv(npiv_result, show_data=True, alpha=0.05, figsize=(8, 6), ucb_result=None):
    """
    Plot NPIV estimates with confidence bands - R style.
    """
    X_eval = npiv_result['X_eval']
    h = npiv_result['h']
    
    if ucb_result is not None:
        h_lower = ucb_result['h_lower']
        h_upper = ucb_result['h_upper']
    else:
        z = norm.ppf(1 - alpha/2)
        h_lower = h - z * npiv_result['se']
        h_upper = h + z * npiv_result['se']
    
    sort_idx = np.argsort(X_eval)
    X_sorted = X_eval[sort_idx]
    h_sorted = h[sort_idx]
    lower_sorted = h_lower[sort_idx]
    upper_sorted = h_upper[sort_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_data:
        ax.scatter(npiv_result['X'], npiv_result['Y'], 
                   s=10, c='gray', alpha=0.4, edgecolors='none')
    
    ax.plot(X_sorted, lower_sorted, 'r--', linewidth=1.5, label='Confidence Band')
    ax.plot(X_sorted, upper_sorted, 'r--', linewidth=1.5)
    ax.plot(X_sorted, h_sorted, 'b-', linewidth=2, label='Estimate')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Function')
    ax.legend(loc='upper right', frameon=True)
    
    return fig, ax


def plot_npiv_deriv(npiv_result, alpha=0.05, figsize=(8, 6), ucb_result=None):
    """
    Plot derivative with confidence bands - R style.
    """
    X_eval = npiv_result['X_eval']
    deriv = npiv_result['deriv']
    
    if ucb_result is not None:
        d_lower = ucb_result['deriv_lower']
        d_upper = ucb_result['deriv_upper']
    else:
        z = norm.ppf(1 - alpha/2)
        d_lower = deriv - z * npiv_result['deriv_se']
        d_upper = deriv + z * npiv_result['deriv_se']
    
    sort_idx = np.argsort(X_eval)
    X_sorted = X_eval[sort_idx]
    deriv_sorted = deriv[sort_idx]
    lower_sorted = d_lower[sort_idx]
    upper_sorted = d_upper[sort_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(X_sorted, lower_sorted, 'r--', linewidth=1.5, label='Confidence Band')
    ax.plot(X_sorted, upper_sorted, 'r--', linewidth=1.5)
    ax.plot(X_sorted, deriv_sorted, 'b-', linewidth=2, label='Estimate')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Derivative')
    ax.legend(loc='upper right', frameon=True)
    
    return fig, ax


def summary_npiv(npiv_result):
    """
    Print summary of NPIV estimation - matches R's summary.npiv().
    """
    n = len(npiv_result['Y'])
    n_eval = len(npiv_result['X_eval'])
    n_basis_x = npiv_result['Psi_x'].shape[1]
    n_basis_w = npiv_result['B_w'].shape[1]
    
    resid = npiv_result['residuals']
    sigma = np.sqrt(npiv_result['sigma2'])
    
    Y = npiv_result['Y']
    Y_mean = np.mean(Y)
    ss_tot = np.sum((Y - Y_mean)**2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot
    
    model_type = "Regression" if npiv_result['is_regression'] else "IV"
    
    print(f"\nNonparametric {model_type} Model")
    print("=" * 50)
    print(f"Training points:    {n}")
    if not npiv_result['trainiseval']:
        print(f"Evaluation points:  {n_eval}")
    print(f"Endogenous vars:    1")
    print()
    print(f"B-spline degree for endogenous predictors:   {npiv_result['J_x_degree']}")
    print(f"B-spline segments for endogenous predictors: {npiv_result['J_x_segments']}")
    print()
    if not npiv_result['is_regression']:
        print(f"B-spline degree for instruments:             {npiv_result['K_w_degree']}")
        print(f"B-spline segments for instruments:           {npiv_result['K_w_segments']}")
        print()
    print(f"Residual Std. Error:    {sigma:.4f}")
    print(f"R-squared:              {r2:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    
    # Load Engel95 data
    data_paths = [
        "Engel95.csv",
        "C:/era_data2/Engel95.csv",
        "../Engel95.csv"
    ]
    
    engel = None
    for path in data_paths:
        try:
            engel = pd.read_csv(path)
            print(f"Loaded data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if engel is None:
        raise FileNotFoundError("Engel95.csv not found")
    
    # Sort by logexp (like R)
    engel = engel.sort_values('logexp').reset_index(drop=True)
    
    Y = engel['food'].values
    X = engel['logexp'].values
    W = engel['logwages'].values
    X_eval = np.linspace(4.5, 6.5, 100)
    
    # NPIV with instrument
    print("\n" + "="*60)
    print("NPIV with Instrument (matching R)")
    print("="*60)
    
    result_iv = npiv_estimate(
        Y, X, W,
        J_x_degree=3,
        J_x_segments=1,
        K_w_degree=4,
        K_w_segments=4,
        knots="uniform",  # R's default
        basis="tensor",
        X_eval=X_eval
    )
    
    summary_npiv(result_iv)
    
    # Compare with R
    print("\nComparison with R:")
    print("Head h (Python):", result_iv['h'][:6])
    print("Head h (R):      [0.2614, 0.2603, 0.2591, 0.2580, 0.2568, 0.2557]")
    print()
    print("Tail h (Python):", result_iv['h'][-6:])
    print("Tail h (R):      [0.1362, 0.1344, 0.1326, 0.1307, 0.1288]")
    
    # UCB (Uniform Confidence Bands) - Same method as R (100% Python)
    print("\n")
    ucb_result = npiv_ucb(result_iv, boot_num=499, seed=None)  # More bootstrap = more stable cv
    
    print(f"\nCritical value (h): {ucb_result['cv']:.4f}")
    print(f"Critical value (deriv): {ucb_result['cv_deriv']:.4f}")
    
    # Plot with UCB
    fig1, ax1 = plot_npiv(result_iv, show_data=True, ucb_result=ucb_result)
    ax1.set_title("Engel Curve - NPIV with Instrument")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Function")
    ax1.set_xlim(4.5, 6.5)
    ax1.set_ylim(0.05, 0.38)
    plt.tight_layout()
    plt.savefig("engel_npiv_iv.png", dpi=150)
    plt.show()
    
    # Derivative with UCB
    fig2, ax2 = plot_npiv_deriv(result_iv, ucb_result=ucb_result)
    ax2.set_title("Engel Curve Derivative - NPIV")
    ax2.set_xlabel("X")
    ax2.set_ylim(-0.35, 0.25)
    plt.tight_layout()
    plt.savefig("engel_npiv_deriv.png", dpi=150)
    plt.show()
    
    print("\nDone!")