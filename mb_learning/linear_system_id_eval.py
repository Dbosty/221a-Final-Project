import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt



def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    A = ckpt["A"]
    B = ckpt["B"]
    X_mean = ckpt["X_mean"]
    X_std = ckpt["X_std"]
    U_mean = ckpt["U_mean"]
    U_std = ckpt["U_std"]
    return A, B, X_mean, X_std, U_mean, U_std


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(xn, mean, std):
    return xn * std + mean



def load_data(csv_path, x_star, u_star):
    df = pd.read_csv(csv_path)

    theta = np.deg2rad(df["pendulum_angle"].values)
    theta_dot = df["pendulum_angle_vel"].values
    phi = np.deg2rad(df["servo_angle"].values)
    phi_dot = df["servo_angle_vel"].values
    u = df["motor_voltage"].values

    X = np.vstack([theta, theta_dot, phi, phi_dot]).T

    X = X[:-1]
    X_next = X[1:]
    U = u[:-1].reshape(-1, 1)

    # linearization
    X -= x_star
    X_next -= x_star
    U -= u_star

    return X, U, X_next


def one_step_error(A, B, X, U, X_next):
    errs = []
    for x, u, xn in zip(X, U, X_next):
        pred = A @ x + B.flatten() * u
        errs.append(np.linalg.norm(pred - xn))
    return np.mean(errs), np.std(errs)


def rollout(A, B, x0, U):
    xs = [x0]
    x = x0.copy()
    for u in U:
        x = A @ x + B.flatten() * u
        xs.append(x.copy())
    return np.array(xs)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python linear_system_id_eval.py <model_ckpt> <csv_path>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    csv_path = sys.argv[2]

    # Same equilibrium as training
    x_star = np.array([np.pi, 0.0, 0.0, 0.0])
    u_star = 0.0

    A, B, X_mean, X_std, U_mean, U_std = load_checkpoint(ckpt_path)
    X, U, X_next = load_data(csv_path, x_star, u_star)


    mean_err, std_err = one_step_error(A, B, X, U, X_next)
    print(f"One-step prediction error: mean={mean_err:.6f}, std={std_err:.6f}")

    T = 5001
    x0 = X[5002]
    U_roll = U[:T]

    X_hat = rollout(A, B, x0, U_roll)
    X_true = X[:T+1]

    error = np.sqrt((X_hat - X_true) ** 2)

    import matplotlib.pyplot as plt

    state_names = ["theta", "theta_dot", "phi", "phi_dot"]

    avg_errors = []
    for ind, e in enumerate(error.T):
        print(f"{state_names[ind]} avg error: {np.mean(e)}")
        avg_errors.append(round(np.mean(e), 3))

    plt.figure(figsize=(12,6))
    for i in range(error.shape[1]):
        plt.plot(error[:,i], label=f"{state_names[i]} error ({round(avg_errors[i], 3)})")
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title(f"Prediction Error for All States")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.show()

    labels = ["theta", "theta_dot", "phi", "phi_dot"]
    time = np.arange(len(X_hat))

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(time, X_true[:, i], label="True")
        plt.plot(time, X_hat[:, i], "--", label="Pred")
        plt.ylabel(labels[i])
        if i == 0:
            plt.legend()
    plt.xlabel("Time step")
    plt.suptitle("Linear Model Rollout vs Ground Truth")
    plt.tight_layout()
    plt.show()

    eigs = np.linalg.eigvals(A)
    print("Eigenvalues of A:")
    for ev in eigs:
        print(f"  {ev:.4f}")
    
    print(f"A: {A}")
    print(f"B: {B}")
