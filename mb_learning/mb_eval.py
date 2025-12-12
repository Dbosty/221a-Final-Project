
import sys
import torch
from mb_train import DynamicsModel, MBLDataset

def eval(data):
    dataset = MBLDataset(f"csvs/iped_test_{data}.csv")
    X_t, Xn_t, U_t = dataset.create_state_vectors()

    bs = 256
    e  = 100
    lr = 4e-3
    checkpoint = torch.load(f"runs/pendulum_model_{data}_{bs}_{lr}_{e}.pth", weights_only=False)

    model = DynamicsModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_mean = checkpoint["X_mean"]
    X_std  = checkpoint["X_std"]
    U_mean = checkpoint["U_mean"]
    U_std  = checkpoint["U_std"]

    loss = torch.nn.MSELoss()

    with torch.no_grad():
        pred = model(X_t, U_t)
        mse = loss(pred, Xn_t).item()

    print(f"Unseen data one-step MSE: {mse:.6f}")


    def rollout(model, X_init, U, steps):
        model.eval()
        x = X_init.clone().unsqueeze(0)      
        xs = [x]

        with torch.no_grad():
            for t in range(steps):
                u = U[t].unsqueeze(0)
                x = model(x, u)
                xs.append(x)

        xs = torch.cat(xs, dim=0)
        return xs


    steps = 300
    pred_traj_norm = rollout(model, X_t[0], U_t, steps)

    # unnormalize
    pred_traj = pred_traj_norm.cpu().numpy() * X_std + X_mean
    true_traj = dataset.X_next[:steps+1]          

    error = pred_traj - true_traj


    import matplotlib.pyplot as plt

    state_names = ["theta", "theta_dot", "phi", "phi_dot"]

    plt.figure(figsize=(12,6))
    for i in range(error.shape[1]):
        plt.plot(error[:,i], label=f"{state_names[i]} error")
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Prediction Error for All States")
    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.show()


first = "75_25"
second = "90_10"

if sys.argv[1] == first:
    eval(first)

if sys.argv[1] == second:
    eval(second)

