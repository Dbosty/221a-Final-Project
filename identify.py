"""
Subspace Identification (SID) Algorithms for Linear System Identification.

Implements three subspace identification methods:
- N4SID: Numerical algorithms for Subspace State Space System IDentification
- MOESP: Multivariable Output-Error State sPace identification
- PARSIM-E: PARsimonious SIM with innovation Estimation (closed-loop capable)

References:
    - Qin, S. J. (2006). An overview of subspace identification.
    - Van Overschee & De Moor (1994). N4SID: Subspace algorithms for the
      identification of combined deterministic-stochastic systems.
    - Verhaegen & Dewilde (1992). Subspace model identification.
    - Qin, Lin & Ljung (2005). A novel subspace identification approach with
      enforced causal models (PARSIM-E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.linalg import lstsq, pinv, svd
from scipy.linalg import qr, sqrtm
from scipy.signal import StateSpace


@dataclass
class IdentificationResult:
    """Result of subspace system identification.

    Attributes:
        A: State transition matrix (n x n).
        B: Input matrix (n x nu).
        C: Output matrix (ny x n).
        D: Feedthrough matrix (ny x nu).
        n: Identified system order.
        K: Kalman gain matrix (n x ny), only for PARSIM-E.
        singular_values: Singular values from SVD (for order selection diagnostics).
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    n: int
    K: np.ndarray | None = None
    singular_values: np.ndarray | None = None

    def to_ss(self, dt: float | bool = True) -> StateSpace:
        """Convert to scipy StateSpace representation.

        Args:
            dt: Sampling time for discrete system. True for unspecified discrete,
                None for continuous (though SID produces discrete models).

        Returns:
            scipy.signal.StateSpace object.
        """
        return StateSpace(self.A, self.B, self.C, self.D, dt=dt)


# =============================================================================
# Utility Functions
# =============================================================================


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D with shape (n_variables, n_samples).

    Args:
        arr: Input array, either 1D (n_samples,) or 2D.

    Returns:
        2D array with shape (n_variables, n_samples).
    """
    arr = np.atleast_2d(arr)
    # Convention: if ambiguous, assume fewer variables than samples
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _blkhank(y: np.ndarray, i: int, j: int) -> np.ndarray:
    """Construct a block Hankel matrix.

    Args:
        y: Data matrix (n_vars x n_samples).
        i: Number of block rows.
        j: Number of columns.

    Returns:
        Block Hankel matrix of shape (n_vars * i, j).

    Raises:
        ValueError: If dimensions are invalid.
    """
    n_vars, n_samples = y.shape

    if i < 1:
        raise ValueError("i must be positive")
    if j < 1:
        raise ValueError("j must be positive")
    if j > n_samples - i + 1:
        raise ValueError(f"j={j} too large for data length {n_samples} with i={i}")

    H = np.zeros((n_vars * i, j))
    for k in range(i):
        H[k * n_vars : (k + 1) * n_vars, :] = y[:, k : k + j]
    return H


def _determine_order(S: np.ndarray, threshold: float) -> int:
    """Determine system order from singular values.

    Args:
        S: Array of singular values (descending order).
        threshold: Cumulative energy threshold (0 < threshold <= 1).

    Returns:
        Estimated system order n.
    """
    cumsum = np.cumsum(S) / np.sum(S)
    n = int(np.searchsorted(cumsum, threshold)) + 1
    return min(n, len(S))


def _ridge_lstsq(
    A: np.ndarray, b: np.ndarray, reg: float = 0.0
) -> np.ndarray:
    """Regularized least squares: solve min ||Ax - b||² + reg*||x||².

    Args:
        A: Design matrix (m x n).
        b: Target matrix (m x p).
        reg: Ridge regularization parameter (λ >= 0).

    Returns:
        Solution x (n x p).
    """
    if reg <= 0.0:
        return lstsq(A, b, rcond=None)[0]
    else:
        # Ridge regression via normal equations: (A'A + λI)x = A'b
        ATA = A.T @ A
        ATb = A.T @ b
        n = ATA.shape[0]
        return np.linalg.solve(ATA + reg * np.eye(n), ATb)


def _extract_ac(
    Ok: np.ndarray, ny: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract A and C matrices from extended observability matrix.

    The observability matrix has structure:
        Ok = [C; CA; CA^2; ...; CA^(f-1)]

    Args:
        Ok: Extended observability matrix (ny*f x n).
        ny: Number of outputs.
        n: System order.

    Returns:
        Tuple (A, C) where A is (n x n) and C is (ny x n).
    """
    C = Ok[:ny, :n]

    # A from shift-invariance: Ok[ny:, :] = Ok[:-ny, :] @ A
    Ok_top = Ok[:-ny, :n]
    Ok_bot = Ok[ny:, :n]
    A = lstsq(Ok_top, Ok_bot, rcond=None)[0]

    return A, C


# =============================================================================
# N4SID Algorithm
# =============================================================================


def _n4sid(
    u: np.ndarray,
    y: np.ndarray,
    horizon: int,
    order: int | None,
    svd_threshold: float,
    past_horizon: int,
    feedthrough: bool = True,
    regularization: float = 0.0,
) -> IdentificationResult:
    """N4SID subspace identification algorithm.

    Based on Van Overschee & De Moor formulation using oblique projection.

    Args:
        u: Input data (nu x N).
        y: Output data (ny x N).
        horizon: Future horizon (number of block rows k).
        order: System order (None for automatic selection).
        svd_threshold: Threshold for automatic order selection.
        past_horizon: Past horizon for Hankel matrices.
        feedthrough: If True, estimate D; if False, force D=0.
        regularization: Ridge regularization parameter for least squares.

    Returns:
        IdentificationResult with identified system matrices.
    """
    nu, N = u.shape
    ny = y.shape[0]
    k = horizon

    # Number of columns in Hankel matrices
    j = N - 2 * k + 1
    if j < 1:
        raise ValueError(f"Insufficient data: need N > 2*horizon, got N={N}, horizon={k}")

    # Build block Hankel matrices with 2k block rows
    U = _blkhank(u, 2 * k, j)
    Y = _blkhank(y, 2 * k, j)

    # Split into past and future
    km = k * nu  # rows in past/future input blocks
    kl = k * ny  # rows in past/future output blocks

    Up = U[:km, :]
    Uf = U[km:, :]
    Yp = Y[:kl, :]
    Yf = Y[kl:, :]

    # Combined past data
    Wp = np.vstack([Up, Yp])

    # LQ decomposition of [Uf; Up; Yp; Yf]
    # In numpy, we compute QR of transpose, then transpose back
    stacked = np.vstack([Uf, Up, Yp, Yf])
    _, R = qr(stacked.T, mode="economic")
    L = R.T

    # Extract L blocks (following MATLAB reference indexing)
    # L11: Uf self-correlation
    L11 = L[:km, :km]
    # L21, L22: Up blocks
    L21 = L[km : 2 * km, :km]
    L22 = L[km : 2 * km, km : 2 * km]
    # L31, L32, L33: Yp blocks
    L31 = L[2 * km : 2 * km + kl, :km]
    L32 = L[2 * km : 2 * km + kl, km : 2 * km]
    L33 = L[2 * km : 2 * km + kl, 2 * km : 2 * km + kl]
    # L41, L42, L43, L44: Yf blocks
    L41 = L[2 * km + kl :, :km]
    L42 = L[2 * km + kl :, km : 2 * km]
    L43 = L[2 * km + kl :, 2 * km : 2 * km + kl]

    # Form R matrices for oblique projection
    R22 = np.block([[L22, np.zeros((km, kl))], [L32, L33]])
    R32 = np.hstack([L42, L43])

    # Oblique projection: xi = R32 @ pinv(R22) @ Wp
    xi = R32 @ pinv(R22) @ Wp

    # SVD of oblique projection
    UU, SS, _ = svd(xi, full_matrices=False)
    S_diag = SS

    # Determine system order
    if order is None:
        n = _determine_order(S_diag, svd_threshold)
    else:
        n = order

    n = min(n, len(S_diag), kl)  # Can't exceed data dimensions

    # Extract observability matrix
    U1 = UU[:, :n]
    S1 = np.diag(S_diag[:n])
    Ok = U1 @ sqrtm(S1).real

    # Extract A, C from observability matrix
    A, C = _extract_ac(Ok, ny, n)

    # Extract B, D from Toeplitz structure of Markov parameters
    R21_full = np.vstack([L21, L31])
    R11 = L11
    TOEP = (L41 - R32 @ pinv(R22) @ R21_full) @ pinv(R11)

    # TOEP contains Markov parameters [D; CB; CAB; CA^2B; ...]
    # Extract D (or force to zero if feedthrough=False)
    if feedthrough:
        D = TOEP[:ny, :nu]
    else:
        D = np.zeros((ny, nu))

    # Extract B from Markov parameters using observability structure
    # G_i = CA^(i-1)B, so [G1; G2; ...] = Ok @ [B, AB, ...]
    # We use a least-squares approach
    n_markov = min(k - 1, 4)  # Use a few Markov parameters
    if n_markov >= 1 and n >= 1:
        # Build Hankel of Markov parameters
        G_list = []
        for i in range(n_markov):
            G_list.append(TOEP[(i + 1) * ny : (i + 2) * ny, :nu])

        # Stack and solve for B
        if len(G_list) >= 1:
            G_stacked = np.vstack(G_list)
            Ok_reduced = Ok[: len(G_list) * ny, :n]
            B = _ridge_lstsq(Ok_reduced, G_stacked, regularization)
        else:
            B = np.zeros((n, nu))
    else:
        B = np.zeros((n, nu))

    return IdentificationResult(
        A=A, B=B, C=C, D=D, n=n, K=None, singular_values=S_diag
    )


# =============================================================================
# MOESP Algorithm
# =============================================================================


def _moesp(
    u: np.ndarray,
    y: np.ndarray,
    horizon: int,
    order: int | None,
    svd_threshold: float,
    past_horizon: int,
    feedthrough: bool = True,
    regularization: float = 0.0,
) -> IdentificationResult:
    """MOESP subspace identification algorithm.

    Based on Verhaegen & Dewilde formulation using orthogonal projection.

    Args:
        u: Input data (nu x N).
        y: Output data (ny x N).
        horizon: Number of block rows R.
        order: System order (None for automatic selection).
        svd_threshold: Threshold for automatic order selection.
        past_horizon: Not used in basic MOESP (kept for interface consistency).
        feedthrough: If True, estimate D; if False, force D=0.
        regularization: Ridge regularization parameter for least squares.

    Returns:
        IdentificationResult with identified system matrices.
    """
    nu, N = u.shape
    ny = y.shape[0]
    R = horizon

    # Number of columns
    j = N - R + 1
    if j < 1:
        raise ValueError(f"Insufficient data: need N >= horizon, got N={N}, horizon={R}")

    # Build block Hankel matrices
    U = _blkhank(u, R, j)
    Y = _blkhank(y, R, j)

    km = R * nu  # rows of U
    kp = R * ny  # rows of Y

    # LQ decomposition of [U; Y]
    stacked = np.vstack([U, Y])
    _, L_full = qr(stacked.T, mode="economic")
    L = L_full.T

    # Extract blocks
    L11 = L[:km, :km]
    L21 = L[km : km + kp, :km]
    L22 = L[km : km + kp, km : km + kp]

    # SVD of L22 (output subspace orthogonal to input)
    UU, SS, _ = svd(L22, full_matrices=False)
    S_diag = SS

    # Determine system order
    if order is None:
        n = _determine_order(S_diag, svd_threshold)
    else:
        n = order

    n = min(n, len(S_diag), kp)

    # Extract observability matrix
    U1 = UU[:, :n]
    S1 = np.diag(S_diag[:n])
    Ok = U1 @ sqrtm(S1).real

    # Extract A, C from observability matrix
    A, C = _extract_ac(Ok, ny, n)

    # Extract B, D via Markov parameter structure
    # Following MOESP formulation: use null space approach
    U2 = UU[:, n:]

    if U2.shape[1] > 0:
        Z = U2.T @ L21 @ pinv(L11)

        # Build least-squares structure for B, D
        XX_list = []
        RR_list = []

        for j_idx in range(R):
            XX_list.append(Z[:, j_idx * nu : (j_idx + 1) * nu])

            # Build Rj matrix with exactly R*ny rows so that the
            # multiplication U2.T @ Rj is dimensionally consistent.
            # Use only (R-1-j_idx) output blocks from Ok to match kp = R*ny.
            if j_idx < R - 1:
                Okj = Ok[: (R - 1 - j_idx) * ny, :]
            else:
                Okj = np.zeros((0, n))

            Rj_top = np.zeros((j_idx * ny, ny + n))
            Rj_mid = np.hstack([np.eye(ny), np.zeros((ny, n))])
            if Okj.shape[0] > 0:
                Rj_bot = np.hstack([np.zeros((Okj.shape[0], ny)), Okj])
            else:
                Rj_bot = np.zeros((0, ny + n))

            Rj = np.vstack([Rj_top, Rj_mid, Rj_bot])
            # Rj now has j_idx*ny + ny + (R-1-j_idx)*ny = R*ny rows, matching kp.
            RR_list.append(U2.T @ Rj)

        XX = np.vstack(XX_list)
        RR = np.vstack(RR_list)

        if feedthrough:
            # Solve for [D; B]
            DB = _ridge_lstsq(RR, XX, regularization)
            D = DB[:ny, :nu]
            B = DB[ny:, :nu]
        else:
            # Force D=0: only solve for B using the B-related part of RR
            # RR has columns [D_cols | B_cols], we use only B_cols
            RR_B = RR[:, ny:]  # Columns for B only
            B = _ridge_lstsq(RR_B, XX, regularization)
            D = np.zeros((ny, nu))
    else:
        # Fallback: simple least-squares from output equation
        D = np.zeros((ny, nu))
        B = np.zeros((n, nu))

    return IdentificationResult(
        A=A, B=B, C=C, D=D, n=n, K=None, singular_values=S_diag
    )


# =============================================================================
# PARSIM-E Algorithm
# =============================================================================


def _parsim_e(
    u: np.ndarray,
    y: np.ndarray,
    horizon: int,
    order: int | None,
    svd_threshold: float,
    past_horizon: int,
    feedthrough: bool = True,
    regularization: float = 0.0,
) -> IdentificationResult:
    """PARSIM-E subspace identification algorithm.

    Parsimonious SIM with innovation Estimation. Suitable for closed-loop
    identification due to row-wise causality enforcement.

    Args:
        u: Input data (nu x N).
        y: Output data (ny x N).
        horizon: Future horizon f.
        order: System order (None for automatic selection).
        svd_threshold: Threshold for automatic order selection.
        past_horizon: Past horizon p for state reconstruction.
        feedthrough: If True, estimate D; if False, force D=0.
        regularization: Ridge regularization parameter for least squares.

    Returns:
        IdentificationResult with identified system matrices including Kalman gain K.
    """
    nu, N = u.shape
    ny = y.shape[0]
    f = horizon
    p = past_horizon

    # Data length check
    total_horizon = f + p
    j = N - total_horizon + 1
    if j < 1:
        raise ValueError(
            f"Insufficient data: need N >= f + p, got N={N}, f={f}, p={p}"
        )

    # Build past data Hankel matrices
    Up = _blkhank(u[:, : N - f + 1], p, j)  # Past inputs
    Yp = _blkhank(y[:, : N - f + 1], p, j)  # Past outputs
    Zp = np.vstack([Up, Yp])  # Combined past instrument

    # Build future output rows
    # Yf[i] corresponds to y_{k+i-1} for row i (1-indexed)
    Yf_full = _blkhank(y[:, p:], f, j)

    # Storage for Gamma*Lz estimates and innovation estimates
    GammaLz_list = []
    H_list = []
    G_tilde_list = []
    E_hat = None  # Accumulated innovation estimates

    for i in range(1, f + 1):
        # Extract row i of future outputs: y_{k+i-1}
        Yfi = Yf_full[(i - 1) * ny : i * ny, :]

        # Build Ui: inputs from k to k+i-1
        # When feedthrough=False and D=0:
        #   - H_{fi} = [CA^{i-2}B ... CB] (no D term)
        #   - So Ui has (i-1) blocks instead of i blocks
        if feedthrough:
            n_input_blocks = i
        else:
            n_input_blocks = i - 1  # Exclude contemporaneous input (D=0)

        if n_input_blocks > 0:
            Ui_rows = []
            for idx in range(n_input_blocks):
                u_row = u[:, p + idx : p + idx + j]
                Ui_rows.append(u_row)
            Ui = np.vstack(Ui_rows)
        else:
            Ui = None

        if i == 1:
            # First row: no past innovation
            if Ui is not None:
                # [Gamma_1 * Lz, H_1] = Yf1 @ pinv([Zp; U1])
                regressor = np.vstack([Zp, Ui])
            else:
                # D=0: Gamma_1 * Lz = Yf1 @ pinv(Zp)
                regressor = Zp

            coeffs = _ridge_lstsq(regressor.T, Yfi.T, regularization).T

            # Split coefficients: [Gamma*Lz | H]
            split_idx = Zp.shape[0]
            GammaLz_i = coeffs[:, :split_idx]
            H_i = coeffs[:, split_idx:] if Ui is not None else np.zeros((ny, 0))

            # Innovation estimate
            E_fi = Yfi - coeffs @ regressor
            E_hat = E_fi

        else:
            # Rows i >= 2: include past innovation estimates
            if Ui is not None:
                regressor = np.vstack([Zp, Ui, E_hat])
            else:
                regressor = np.vstack([Zp, E_hat])

            coeffs = _ridge_lstsq(regressor.T, Yfi.T, regularization).T

            # Split coefficients: [Gamma*Lz | H | G_tilde]
            split1 = Zp.shape[0]
            if Ui is not None:
                split2 = split1 + Ui.shape[0]
                GammaLz_i = coeffs[:, :split1]
                H_i = coeffs[:, split1:split2]
                G_tilde_i = coeffs[:, split2:]
            else:
                GammaLz_i = coeffs[:, :split1]
                H_i = np.zeros((ny, 0))
                G_tilde_i = coeffs[:, split1:]

            # Innovation estimate for this row
            E_fi = Yfi - coeffs @ regressor

            # Accumulate innovation
            E_hat = np.vstack([E_hat, E_fi])
            G_tilde_list.append(G_tilde_i)

        GammaLz_list.append(GammaLz_i)
        H_list.append(H_i)

    # Stack all Gamma*Lz estimates
    GammaLz = np.vstack(GammaLz_list)

    # SVD to extract observability matrix
    UU, SS, _ = svd(GammaLz, full_matrices=False)
    S_diag = SS

    # Determine system order
    if order is None:
        n = _determine_order(S_diag, svd_threshold)
    else:
        n = order

    n = min(n, len(S_diag), f * ny)

    # Extract observability matrix
    U1 = UU[:, :n]
    S1 = np.diag(S_diag[:n])
    Ok = U1 @ sqrtm(S1).real

    # Extract A, C
    A, C = _extract_ac(Ok, ny, n)

    # Extract D from H_1
    # When feedthrough=True: H_1 = [D], so D = H_list[0]
    # When feedthrough=False: D = 0 (was not estimated)
    if feedthrough and H_list[0].shape[1] >= nu:
        D = H_list[0][:, -nu:]
    else:
        D = np.zeros((ny, nu))

    # Extract B from the H matrices structure
    # When feedthrough=True: H_i = [CA^{i-2}B, ..., CB, D], CB is first nu cols of H_2
    # When feedthrough=False: H_i = [CA^{i-2}B, ..., CB], CB is first nu cols of H_2
    if len(H_list) >= 2 and H_list[1].shape[1] >= nu:
        CB = H_list[1][:, :nu]
        B = _ridge_lstsq(C, CB, regularization)
    else:
        B = np.zeros((n, nu))

    # Extract Kalman gain K from G_tilde matrices
    # G_tilde_i contains [CA^{i-2}K, ..., CK] for rows of length (i-1)*ny
    # For i=2: G_tilde_2 = [CK], so CK is in the last ny columns
    K = None
    if len(G_tilde_list) >= 1 and G_tilde_list[0].shape[1] >= ny:
        CK = G_tilde_list[0][:, -ny:]
        K = _ridge_lstsq(C, CK, regularization)

    return IdentificationResult(
        A=A, B=B, C=C, D=D, n=n, K=K, singular_values=S_diag
    )


# =============================================================================
# Main Interface
# =============================================================================


def subspace_id(
    u: np.ndarray,
    y: np.ndarray,
    method: Literal["n4sid", "moesp", "parsim_e"] = "n4sid",
    horizon: int = 15,
    order: int | None = None,
    svd_threshold: float = 0.85,
    past_horizon: int | None = None,
    feedthrough: bool = True,
    regularization: float = 0.0,
    normalize: bool = True,
) -> IdentificationResult:
    """Unified interface for subspace system identification.

    Identifies a discrete-time linear state-space model from input-output data:
        x[k+1] = A @ x[k] + B @ u[k]
        y[k]   = C @ x[k] + D @ u[k]

    Args:
        u: Input data. Shape (N,) for SISO or (nu, N) for MIMO.
        y: Output data. Shape (N,) for SISO or (ny, N) for MIMO.
        method: Identification algorithm:
            - "n4sid": N4SID with oblique projection (default).
            - "moesp": MOESP with orthogonal projection.
            - "parsim_e": PARSIM-E with innovation estimation (closed-loop capable).
        horizon: Future horizon / number of block rows. Typically 10-30.
            Must be greater than expected system order.
        order: System order n. If None, automatically selected via SVD threshold.
        svd_threshold: Cumulative singular value energy threshold for automatic
            order selection. Range (0, 1], default 0.85. Higher values yield
            higher order models.
        past_horizon: Past horizon p for state reconstruction. Defaults to
            `horizon` if not specified. Only affects PARSIM-E significantly.
        feedthrough: If True (default), estimate the D matrix. If False, force
            D=0 in the identification. This constraint is applied during the
            Markov parameter estimation, not as a post-processing step.
        regularization: Ridge regularization parameter (λ >= 0) for least-squares
            operations. Default 0.0 (no regularization). Small positive values
            (e.g., 1e-6 to 1e-3) can improve numerical stability for
            ill-conditioned problems.
        normalize: If True (default), normalize input/output data to zero mean
            and unit variance before identification, then transform the results
            back to the original scale. This improves numerical conditioning.

    Returns:
        IdentificationResult containing:
            - A, B, C, D: State-space matrices
            - n: Identified system order
            - K: Kalman gain (only for PARSIM-E)
            - singular_values: SVD singular values for diagnostics

    Raises:
        ValueError: If inputs are invalid or data is insufficient.

    Example:
        >>> import numpy as np
        >>> # Generate test data from known system
        >>> A_true = np.array([[0.9, 0.1], [-0.1, 0.8]])
        >>> B_true = np.array([[1.0], [0.5]])
        >>> C_true = np.array([[1.0, 0.0]])
        >>> D_true = np.array([[0.0]])
        >>> N = 1000
        >>> u = np.random.randn(N)
        >>> x = np.zeros((2, N))
        >>> y = np.zeros(N)
        >>> for k in range(N - 1):
        ...     y[k] = C_true @ x[:, k] + D_true * u[k]
        ...     x[:, k + 1] = A_true @ x[:, k] + B_true * u[k]
        >>> # Identify system with D=0 constraint
        >>> result = subspace_id(u, y, method="n4sid", horizon=20, order=2, feedthrough=False)
        >>> ss = result.to_ss()
    """
    # Validate and reshape inputs
    u = _ensure_2d(np.asarray(u, dtype=float))
    y = _ensure_2d(np.asarray(y, dtype=float))

    nu, N_u = u.shape
    ny, N_y = y.shape

    if N_u != N_y:
        raise ValueError(f"Input and output must have same length: {N_u} != {N_y}")

    N = N_u

    # Validate parameters
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if svd_threshold <= 0 or svd_threshold > 1:
        raise ValueError("svd_threshold must be in (0, 1]")
    if order is not None and order < 1:
        raise ValueError("order must be at least 1")

    # Set default past horizon
    if past_horizon is None:
        past_horizon = horizon

    # Validate regularization
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    # Normalize data for better conditioning
    if normalize:
        u_mean = u.mean(axis=1, keepdims=True)
        u_std = u.std(axis=1, keepdims=True)
        u_std = np.where(u_std < 1e-10, 1.0, u_std)  # Avoid division by zero
        u_norm = (u - u_mean) / u_std

        y_mean = y.mean(axis=1, keepdims=True)
        y_std = y.std(axis=1, keepdims=True)
        y_std = np.where(y_std < 1e-10, 1.0, y_std)  # Avoid division by zero
        y_norm = (y - y_mean) / y_std
    else:
        u_norm, y_norm = u, y
        u_std = np.ones((nu, 1))
        y_std = np.ones((ny, 1))

    # Dispatch to algorithm (using normalized data)
    method_lower = method.lower()
    if method_lower == "n4sid":
        result = _n4sid(
            u_norm, y_norm, horizon, order, svd_threshold, past_horizon,
            feedthrough, regularization
        )
    elif method_lower == "moesp":
        result = _moesp(
            u_norm, y_norm, horizon, order, svd_threshold, past_horizon,
            feedthrough, regularization
        )
    elif method_lower == "parsim_e":
        result = _parsim_e(
            u_norm, y_norm, horizon, order, svd_threshold, past_horizon,
            feedthrough, regularization
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'n4sid', 'moesp', or 'parsim_e'")

    # Transform results back to original scale
    # State-space transformation for normalized data:
    #   u_norm = (u - u_mean) / u_std,  y_norm = (y - y_mean) / y_std
    # Identified: y_norm = C_n @ x + D_n @ u_norm
    # Original:   y = y_std * (C_n @ x + D_n @ (u - u_mean)/u_std) + y_mean
    #               = (y_std * C_n) @ x + (y_std * D_n / u_std) @ u + offset
    # So: A unchanged, B /= u_std, C *= y_std, D *= y_std / u_std, K /= y_std
    if normalize:
        # u_std is (nu, 1), y_std is (ny, 1)
        # B is (n, nu): divide each column j by u_std[j]
        B_new = result.B / u_std.T
        # C is (ny, n): multiply each row i by y_std[i]
        C_new = result.C * y_std
        # D is (ny, nu): multiply row i by y_std[i], divide col j by u_std[j]
        D_new = result.D * y_std / u_std.T
        # K is (n, ny): divide each column j by y_std[j]
        K_new = result.K / y_std.T if result.K is not None else None

        result = IdentificationResult(
            A=result.A,
            B=B_new,
            C=C_new,
            D=D_new,
            n=result.n,
            K=K_new,
            singular_values=result.singular_values,
        )

    return result
