from scipy.io import loadmat
import pandas as pd
import numpy as np

start = 43
end = 62
runs = np.arange(start, end+1)
dfs = []
for model in runs:
    mat = loadmat(f"matlab_data/episode_{model}.mat")
    data = {}
    for col in list(mat.keys())[3:]:
        for d in range(5001):
            time = 0
            if col not in data:
                data['seed'] = [model]
                data['time'] = [mat[col].squeeze().tolist()[0][d][0]]
                data[col] = [mat[col].squeeze().tolist()[1][0][0][0][d][0]]
            else:
                data['seed'].append(model)
                data['time'].append(mat[col].squeeze().tolist()[0][d][0])
                data[col].append(mat[col].squeeze().tolist()[1][0][0][0][d][0])

    temp_df = pd.DataFrame(data)
    temp_df['pendulum_angle'] = temp_df['pendulum_angle'] * (180 / np.pi)
    dfs.append(temp_df)
    dfs


df = pd.concat(dfs, ignore_index=True)


dtheta, dphi = 0.002, 0.002
theta = df["pendulum_angle"].values * (np.pi / 180)
phi = df["servo_angle"].values * (np.pi / 180)
times = df["time"].values

thetas = []
phis = []

for i in range(len(df) - 1):               
    t = times[i]

    if t == 20.0:
        theta_run, phi_run = [0.0], [0.0]                        

    theta_vel = (theta[i+1] - theta[i]) / dtheta
    phi_vel = (phi[i+1] - phi[i]) / dphi

    theta_run.append(theta_vel)
    phi_run.append(phi_vel)

    if t == 29.998:
        thetas.append(theta_run)
        phis.append(phi_run)
        theta_run, phi_run = [], []

all_thetas, all_phis = [], []
for t, p in zip(thetas, phis):
    for th, ph in zip(t, p):
        all_thetas.append(th)
        all_phis.append(ph)

df["pendulum_angle_vel"] = all_thetas
df["servo_angle_vel"] = all_phis
df["pendulum_angle (rad)"] = theta 
df["servo_angle (rad)"] = phi 


df.to_csv("inv_pend_excite_data.csv")

def get_train_test_data(data):
    """
    Splits the data into a train and test split. 

    data: Data to input 
    train_amt: Amount to put into train
    """
    train = data[data["seed"] <= 57]
    test = data[data["seed"] > 57]

    return train, test
    
train, test = get_train_test_data(df, 0.90)

train.to_csv("iped_train1.csv")
test.to_csv("iped_test1.csv")