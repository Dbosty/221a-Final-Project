import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import cont2discrete
from tqdm import tqdm


def _add_identify_to_path() -> None:
    """Add the directory containing identify.py to sys.path."""
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    identify_dir = project_root / "221a-Final-Project"
    if str(identify_dir) not in sys.path:
        sys.path.insert(0, str(identify_dir))


def _simple_yaml_dump(data: dict, indent: int = 0) -> str:
    """Very small YAML emitter for nested dicts with scalar leaves."""
    lines: list[str] = []
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_simple_yaml_dump(value, indent + 1))
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def _simulate_outputs(u: np.ndarray, result, x0: np.ndarray | None = None) -> np.ndarray:
    """Simulate output sequence y_hat for given input using identified model.

    Uses a simple forward simulation and clips state/output magnitudes to
    avoid numerical overflow for unstable identified models.
    """
    A, B, C, D = result.A, result.B, result.C, result.D

    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(1, -1)

    _, N = u.shape
    n = A.shape[0]
    ny = C.shape[0]

    if x0 is None:
        x = np.zeros((n, 1))
    else:
        x = np.asarray(x0, dtype=float).reshape(n, 1)

    y_hat = np.zeros((ny, N))

    x_clip = 1e6
    y_clip = 1e6

    for k in range(N):
        uk = u[:, k : k + 1]
        y_k = C @ x + D @ uk
        y_k = np.clip(y_k, -y_clip, y_clip)
        y_hat[:, k : k + 1] = y_k

        x = A @ x + B @ uk
        x = np.clip(x, -x_clip, x_clip)

    return y_hat


def _trig_transform(y_arr: np.ndarray, column_names: list[str]):
    """Convert angle rows into sin/cos pairs in radians."""
    features = []
    names: list[str] = []
    for row, name in zip(y_arr, column_names):
        radians = np.deg2rad(row)
        sin_row = np.sin(radians)
        cos_row = np.cos(radians)
        features.append(sin_row)
        features.append(cos_row)
        names.extend([f"{name}_sin", f"{name}_cos"])
    return np.vstack(features), names


def _format_outputs(y_angles: np.ndarray, output_cols: list[str], use_trig: bool):
    if use_trig:
        y_trig, _ = _trig_transform(y_angles, output_cols)
        return y_trig
    return y_angles


def _estimate_sample_time(df: pd.DataFrame) -> float:
    diffs: list[float] = []
    for seed in sorted(df["seed"].unique()):
        times = df[df["seed"] == seed]["time"].to_numpy()
        if len(times) >= 2:
            dt = np.diff(times)
            dt = dt[dt > 0]
            diffs.extend(dt.tolist())
    if not diffs:
        raise RuntimeError("Unable to estimate sample time from data.")
    return float(np.median(diffs))


def _quanser_dynamics_state(x: np.ndarray, tau: float) -> np.ndarray:
    m1 = 0.097
    Lp = 0.200025
    l1 = 0.163195
    J1 = 3.234e-4
    b2 = 0.0024
    J0 = 0.00322176
    b1 = 0.015
    L0 = 0.085
    g = 9.81

    th1, w1, th2, w2 = x
    M11 = (
        J0 + 0.25 * m1 * L0**2 + 0.25 * m1 * Lp**2 - 0.5 * m1 * Lp * L0 * np.cos(th2)
    )
    M12 = -0.5 * m1 * Lp * L0 * np.cos(th2)
    M21 = M12
    M22 = J1 + 0.25 * m1 * Lp**2
    M = np.array([[M11, M12], [M21, M22]])

    N1 = (
        0.5 * m1 * Lp * np.sin(th2) * np.cos(th2) * w1 * w2
        + 0.5 * m1 * Lp * L0 * np.sin(th2) * w2**2
    )
    N2 = (
        -0.5 * m1 * Lp**2 * np.cos(th2) * np.sin(th2) * w1**2
        - 0.5 * m1 * Lp * g * np.sin(th2)
    )
    Q1 = tau - b1 * w1
    Q2 = -b2 * w2
    rhs = np.array([Q1 - N1, Q2 - N2])
    qdd = np.linalg.solve(M, rhs)
    return np.array([w1, qdd[0], w2, qdd[1]])


def _linearize_quanser() -> tuple[np.ndarray, np.ndarray]:
    x_eq = np.zeros(4)
    tau_eq = 0.0
    n = 4
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    eps = 1e-6
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        f_plus = _quanser_dynamics_state(x_eq + dx, tau_eq)
        f_minus = _quanser_dynamics_state(x_eq - dx, tau_eq)
        A[:, j] = (f_plus - f_minus) / (2 * eps)
    du = 1e-6
    f_plus = _quanser_dynamics_state(x_eq, tau_eq + du)
    f_minus = _quanser_dynamics_state(x_eq, tau_eq - du)
    B[:, 0] = (f_plus - f_minus) / (2 * du)
    return A, B


def _build_linear_baseline(dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_c, B_c = _linearize_quanser()
    C_c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    D_c = np.zeros((2, 1))
    Ad, Bd, Cd, Dd, _ = cont2discrete((A_c, B_c, C_c, D_c), dt)
    return Ad, Bd, Cd, Dd


def _simulate_linear_baseline_episode(
    u_seg: np.ndarray, Ad: np.ndarray, Bd: np.ndarray, Cd: np.ndarray, Dd: np.ndarray
) -> np.ndarray:
    n = Ad.shape[0]
    x = np.zeros((n, 1))
    N = u_seg.shape[1]
    y = np.zeros((Cd.shape[0], N))
    for k in range(N):
        uk = u_seg[:, k : k + 1]
        y[:, k : k + 1] = Cd @ x + Dd @ uk
        x = Ad @ x + Bd @ uk
    return y


def _compute_error_stats(err_train: np.ndarray, err_test: np.ndarray, feature_names: list[str]):
    method_stats = {"train": {}, "test": {}}
    for idx, col in enumerate(feature_names):
        e_tr = err_train[idx, :]
        e_te = err_test[idx, :]
        method_stats["train"][col] = {
            "mse": float(np.mean(e_tr**2)),
            "std": float(np.std(e_tr)),
        }
        method_stats["test"][col] = {
            "mse": float(np.mean(e_te**2)),
            "std": float(np.std(e_te)),
        }
    return method_stats


def _build_episode_batches(
    df_segment: pd.DataFrame,
    seed_list: list[int],
    input_col: str,
    output_cols: list[str],
    use_trig: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[str]]:
    """Return per-episode input/output arrays and metadata."""
    u_batch: list[np.ndarray] = []
    y_batch: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for seed in seed_list:
        seg = df_segment[df_segment["seed"] == seed].sort_values(["time"])
        u_arr = seg[input_col].to_numpy().reshape(1, -1)
        y_arr = seg[output_cols].to_numpy().T
        if use_trig:
            y_arr, names = _trig_transform(y_arr, output_cols)
            if feature_names is None:
                feature_names = names
        else:
            if feature_names is None:
                feature_names = output_cols
        u_batch.append(u_arr)
        y_batch.append(y_arr)
    y_concat = np.hstack(y_batch)
    return u_batch, y_batch, y_concat, feature_names or output_cols


def main() -> None:
    _add_identify_to_path()
    from identify import subspace_id  # type: ignore

    data_path = Path(__file__).with_name("inv_pend_excite_data.csv")
    df = pd.read_csv(data_path)

    # Define episodes by unique seed; sort so episode index is deterministic.
    seeds = sorted(df["seed"].unique())
    if len(seeds) < 19:
        raise RuntimeError("Expected at least 19 episodes in the CSV.")

    # Episode indices are 1-based in the problem statement.
    train_seeds = seeds[0:15]   # episodes 1–15
    test_seeds = seeds[15:19]   # episodes 16–19

    train_df = df[df["seed"].isin(train_seeds)].copy()
    test_df = df[df["seed"].isin(test_seeds)].copy()

    # Sort by seed and time to ensure consistent ordering.
    train_df = train_df.sort_values(["seed", "time"])
    test_df = test_df.sort_values(["seed", "time"])

    input_col = "motor_voltage"
    output_cols = ["servo_angle", "pendulum_angle"]
    use_trig_transform = True
    sample_time = _estimate_sample_time(df)
    Ad_lin, Bd_lin, Cd_lin, Dd_lin = _build_linear_baseline(sample_time)

    u_train_batch, y_train_batch, y_train, feature_names = _build_episode_batches(
        train_df, list(train_seeds), input_col, output_cols, use_trig_transform
    )
    u_test_batch, y_test_batch, y_test, _ = _build_episode_batches(
        test_df, list(test_seeds), input_col, output_cols, use_trig_transform
    )

    methods = ["n4sid", "moesp", "parsim_e"]

    stats: dict = {}

    for method in tqdm(methods, desc="Identification", unit="method"):
        try:
            result = subspace_id(
                u_train_batch,
                y_train_batch,
                method=method,
                horizon=50,
                order=10, #8
                past_horizon=50,
                feedthrough=False,
                svd_threshold=0.9,
                regularization=0.2, #0.01
            )
        except Exception as exc:  # noqa: BLE001
            # Record failure reason (without problematic YAML characters).
            stats[method] = {
                "error": f"{type(exc).__name__}"
            }
            continue

        # Simulate each episode separately to reset state and reduce
        # long-horizon drift for unstable models.
        def simulate_batches(u_batch: list[np.ndarray]) -> np.ndarray:
            yhats = [_simulate_outputs(u_seg, result) for u_seg in u_batch]
            return np.hstack(yhats)

        yhat_train = simulate_batches(u_train_batch)
        yhat_test = simulate_batches(u_test_batch)

        err_train = y_train - yhat_train
        err_test = y_test - yhat_test
        stats[method] = _compute_error_stats(err_train, err_test, feature_names)

    def simulate_linear_batches(u_batch: list[np.ndarray]) -> np.ndarray:
        yhats = []
        for u_seg in u_batch:
            angles = _simulate_linear_baseline_episode(u_seg, Ad_lin, Bd_lin, Cd_lin, Dd_lin)
            yhats.append(_format_outputs(angles, output_cols, use_trig_transform))
        return np.hstack(yhats)

    yhat_train_lin = simulate_linear_batches(u_train_batch)
    yhat_test_lin = simulate_linear_batches(u_test_batch)
    stats["linear_baseline"] = _compute_error_stats(
        y_train - yhat_train_lin, y_test - yhat_test_lin, feature_names
    )

    yaml_str = _simple_yaml_dump(stats)
    out_path = Path(__file__).with_name("identification_error_stats.yaml")
    out_path.write_text(yaml_str)


if __name__ == "__main__":
    main()
