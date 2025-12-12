import sys
from pathlib import Path

import numpy as np
import pandas as pd


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

    u_train = train_df[input_col].to_numpy().reshape(1, -1)
    y_train = train_df[output_cols].to_numpy().T

    u_test = test_df[input_col].to_numpy().reshape(1, -1)
    y_test = test_df[output_cols].to_numpy().T

    methods = ["n4sid", "moesp", "parsim_e"]

    stats: dict = {}

    for method in methods:
        try:
            result = subspace_id(
                u_train,
                y_train,
                method=method,
                horizon=20,
                order=8,
                past_horizon=20,
                feedthrough=False,
                svd_threshold=0.1,
                regularization=1e-3,
            )
        except Exception as exc:  # noqa: BLE001
            # Record failure reason (without problematic YAML characters).
            stats[method] = {
                "error": f"{type(exc).__name__}"
            }
            continue

        # Simulate each episode separately to reset state and reduce
        # long-horizon drift for unstable models.
        def simulate_by_seeds(df_segment: pd.DataFrame, seed_list: list[int]) -> np.ndarray:
            yhats: list[np.ndarray] = []
            for s in seed_list:
                seg = df_segment[df_segment["seed"] == s].sort_values(["time"])
                u_seg = seg[input_col].to_numpy().reshape(1, -1)
                yhat_seg = _simulate_outputs(u_seg, result)
                yhats.append(yhat_seg)
            return np.concatenate(yhats, axis=1)

        yhat_train = simulate_by_seeds(train_df, list(train_seeds))
        yhat_test = simulate_by_seeds(test_df, list(test_seeds))

        err_train = y_train - yhat_train
        err_test = y_test - yhat_test

        method_stats = {"train": {}, "test": {}}

        for idx, col in enumerate(output_cols):
            e_tr = err_train[idx, :]
            e_te = err_test[idx, :]
            method_stats["train"][col] = {
                "mean": float(np.mean(e_tr)),
                "std": float(np.std(e_tr)),
            }
            method_stats["test"][col] = {
                "mean": float(np.mean(e_te)),
                "std": float(np.std(e_te)),
            }

        stats[method] = method_stats

    yaml_str = _simple_yaml_dump(stats)
    out_path = Path(__file__).with_name("identification_error_stats.yaml")
    out_path.write_text(yaml_str)


if __name__ == "__main__":
    main()
