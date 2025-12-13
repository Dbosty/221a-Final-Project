import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MBLDataset:

    def __init__(self, file_path):

        self.file_path = file_path

        self.get_states_from_data()

    def get_states_from_data(self):
        """Reads in data as a file path and returns states."""
        df = pd.read_csv(self.file_path)

        theta_deg = df["pendulum_angle"].values
        phi_deg   = df["servo_angle"].values

        self.theta = np.deg2rad(theta_deg)
        self.phi   = np.deg2rad(phi_deg)

        self.u = df["motor_voltage"].values

        self.theta_dot = df["pendulum_angle_vel"]
        self.phi_dot   = df["servo_angle_vel"]

    
    def create_state_vectors(self):
        """Creates the normalized state vectors and converts them into torch tensors."""
        # Build state vectors
        self.X = np.vstack([self.theta, self.theta_dot, self.phi, self.phi_dot]).T

        self.X_next = self.X[1:]
        self.X      = self.X[:-1]
        self.u      = self.u[:-1].reshape(-1, 1)

        self.X_mean = self.X.mean(axis=0)
        self.X_std  = self.X.std(axis=0) + 1e-6

        self.U_mean = self.u.mean(axis=0)
        self.U_std  = self.u.std(axis=0) + 1e-6

        self.X_norm      = (self.X - self.X_mean) / self.X_std
        self.X_next_norm = (self.X_next - self.X_mean) / self.X_std
        self.U_norm      = (self.u - self.U_mean) / self.U_std

        X_t      = torch.tensor(self.X_norm,      dtype=torch.float32)
        Xn_t     = torch.tensor(self.X_next_norm, dtype=torch.float32)
        U_t      = torch.tensor(self.U_norm,      dtype=torch.float32)

        return X_t, Xn_t, U_t


    def create_train_split(self):
        """Creates train-test split on state / input data"""
        X_t, Xn_t, U_t = self.create_state_vectors()
        N = len(X_t)
        split = int(0.8 * N)

        X_train, U_train, Xn_train = X_t[:split], U_t[:split], Xn_t[:split]
        X_val,   U_val,   Xn_val   = X_t[split:], U_t[split:], Xn_t[split:]

        return X_train, X_val, U_train, U_val, Xn_train, Xn_val


class DynamicsModel(nn.Module):
    """Simple Learning-Based Dynamics Model"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x, u):
        x = self.net(torch.cat([x, u], dim=1))  # Adding in the dx helped a lot with training!
        return x

class LinearDynamicsModel(nn.Module):
    """x_{t+1} = A x_t + B u_t"""
    def __init__(self, n=4, m=1):
        super().__init__()
        self.linear = nn.Linear(n + m, n, bias=False)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        return self.linear(xu)

    def get_matrices(self):
        W = self.linear.weight.data.cpu().numpy()
        A = W[:, :4]
        B = W[:, 4:]
        return A, B

class SystemIDTrainer:
    def __init__(self, 
                 model,
                 optimizer,
                 loss,
                 file_path,
                 lr=1e-3, 
                 batch_size=256, 
                 epochs=100):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr
        self.bs = batch_size
        self.e = epochs

        self.file_path = file_path

        self.dataset = MBLDataset(f"csvs/iped_train_{self.file_path}.csv")
        self.X_train, self.X_val, self.U_train, \
            self.U_val, self.Xn_train, self.Xn_val = self.dataset.create_train_split()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def batches(self, X, U, Y):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), self.bs):
            j = idx[i:i+self.bs]
            yield X[j], U[j], Y[j]

    def train(self):
        for ep in range(self.e):
            self.model.train()
            train_losses = []

            for xb, ub, yb in self.batches(self.X_train, self.U_train, self.Xn_train):
                pred = self.model(xb, ub)
                loss = self.loss(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(self.X_val, self.U_val)
                val_loss = self.loss(val_pred, self.Xn_val).item()

            print(f"Epoch {ep+1:03d} | Train: {np.mean(train_losses):.6f} | Val: {val_loss:.6f}")

    
        A, B = self.model.get_matrices()
        torch.save({
        "model_state_dict": self.model.state_dict(),
        "A": A,
        "B": B,
        "X_mean": self.dataset.X_mean,
        "X_std": self.dataset.X_std,
        "U_mean": self.dataset.U_mean,
        "U_std": self.dataset.U_std,
        }, f"runs/lin_model_{self.file_path}_{self.bs}_{self.lr}_{self.e}.pth")

class MBLTrainer:
    """Model-based Learning Trainer"""
    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 file_path,
                 lr=4e-3,
                 batch_size=256,
                 epochs=100
                ):

        self.model     = model
        self.optimizer = optimizer
        self.loss      = loss
        self.lr        = lr
        self.bs        = batch_size
        self.e         = epochs
        self.file_path = file_path

        self.dataset = MBLDataset(f"csvs/iped_train_{self.file_path}.csv")
        self.X_train, self.X_val, self.U_train, \
            self.U_val, self.Xn_train, self.Xn_val = self.dataset.create_train_split()
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    @staticmethod
    def get_batches(X, U, Y, bs):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            j = idx[i:i+bs]
            yield X[j], U[j], Y[j]

    def train(self):

        for epoch in range(self.e):

            # Begin training here 
            self.model.train()
            train_losses = []

            for Xb, Ub, Yb in self.get_batches(self.X_train, self.U_train, self.Xn_train, self.bs):
                pred = self.model(Xb, Ub)
                loss = self.loss(pred, Yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(self.X_val, self.U_val)
                val_loss = self.loss(val_pred, self.Xn_val).item()

            print(f"Epoch {epoch+1}/{self.e}   "
                f"Train Loss: {np.mean(train_losses):.6f}   "
                f"Val Loss: {val_loss:.6f}")



        torch.save({
            "model_state_dict": self.model.state_dict(),
            "X_mean": self.dataset.X_mean,
            "X_std": self.dataset.X_std,
            "U_mean": self.dataset.U_mean,
            "U_std": self.dataset.U_std
        }, f"runs/pendulum_model_{self.file_path}_{self.bs}_{self.lr}_{self.e}.pth")
    
        return self.model


file_path = sys.argv[1]
if sys.argv[2] == "NonLin":
    lr        = 4e-3
    model     = DynamicsModel()
    loss      = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    trainer = MBLTrainer(model, optimizer, loss, file_path)
if sys.argv[2] == "Lin":
    lr=1e-3
    model     = LinearDynamicsModel()
    loss      = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    trainer   = SystemIDTrainer(model, optimizer, loss, file_path)


if __name__ == "__main__":
    
    trainer.train()