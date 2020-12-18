import sys
sys.path.append("..")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.optim as optim

import mbrl.models as models
import mbrl.replay_buffer as replay_buffer

device = torch.device("cuda:0")

mpl.rcParams['figure.facecolor'] = 'white'

def create_data():

    x_data = np.linspace(-12, 12, 10000)
    y_data = np.sin(x_data)

    train_size = 2000
    val_size = 200
    x_train = np.zeros(2 * train_size)
    y_train = np.zeros(2 * train_size)
    x_val = np.zeros(2 * val_size)
    y_val = np.zeros(2 * val_size)

    # Half with lower noise
    train_val_idx_1 = np.random.choice(list(range(1200, 3500)), 
                                       size=train_size + val_size, 
                                       replace=False)
    mag = 0.05
    x_train[:train_size] = x_data[train_val_idx_1[:train_size]]
    y_train[:train_size] = y_data[train_val_idx_1[:train_size]] + mag * np.random.randn(train_size)
    x_val[:val_size] = x_data[train_val_idx_1[train_size:]]
    y_val[:val_size] = y_data[train_val_idx_1[train_size:]] + mag * np.random.randn(val_size)

    # Half with higher noise
    train_val_idx_2 = np.random.choice(list(range(6500, 8800)), 
                                       size=train_size + val_size, 
                                       replace=False)
    mag = 0.20
    x_train[train_size:] = x_data[train_val_idx_2[:train_size]]
    y_train[train_size:] = y_data[train_val_idx_2[:train_size]] + mag * np.random.randn(train_size)
    x_val[val_size:] = x_data[train_val_idx_2[train_size:]]
    y_val[val_size:] = y_data[train_val_idx_2[train_size:]] + mag * np.random.randn(val_size)

    return x_data, y_data, x_train, y_train, x_val, y_val

def make_dataset(x_train, y_train, x_val, y_val, num_members):
    dataset_train = replay_buffer.BootstrapReplayBuffer(len(x_train), 2048, num_members, (1,), (0,))
    for i in range(len(x_train)):
        dataset_train.add(x_train[i], 0, y_train[i], 0, False)


    dataset_val = replay_buffer.IterableReplayBuffer(len(x_val), 256, (1,), (0,))
    for i in range(len(x_val)):
        dataset_val.add(x_val[i], 0, y_val[i], 0, False)

    return dataset_train, dataset_val

def get_config():
    return omegaconf.OmegaConf.create({
        "_target_": "mbrl.models.GaussianMLP",
        "device": "cuda:0",
        "in_size": 1,
        "out_size": 1,
        "num_layers": 3,
        "hid_size": 32,
        "use_silu": False
    })

def train(ensemble, dataset_train, dataset_val):
    wrapper = models.DynamicsModelWrapper(ensemble, target_is_delta=False, normalize=True, learned_rewards=False)

    trainer = models.DynamicsModelTrainer(wrapper, dataset_train, dataset_val=dataset_val, optim_lr=0.003, weight_decay=5e-5)
    train_losses, val_losses = trainer.train(num_epochs=500, patience=100)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].plot(train_losses)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train loss (gaussian nll)")
    ax[1].plot(val_losses)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val loss (mse)")
    plt.show()

def visualize_results(ensemble, x_data, y_data, x_train, y_train):
    x_tensor = torch.from_numpy(x_data).unsqueeze(1).float().to(device)

    with torch.no_grad():
        y_pred, y_pred_logvar = ensemble(x_tensor)
        y_pred = y_pred[..., 0]
        y_pred_logvar = y_pred_logvar[..., 0]
    y_var_epi = y_pred.var(dim=0).cpu().numpy()
    y_var = y_pred_logvar.exp()
    y_pred = y_pred.mean(dim=0).cpu().numpy()
    y_var_ale = y_var.mean(dim=0).cpu().numpy()

    y_std = np.sqrt(y_var_epi + y_var_ale)
    plt.figure(figsize=(16, 8))
    plt.plot(x_data, y_data, 'r')
    plt.plot(x_train, y_train, '.', markersize=0.9)
    plt.plot(x_data, y_pred, 'b-', markersize=4)
    plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
    plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
    plt.axis([-12, 12, -2.5, 2.5])
    plt.show()

def main():
    x_data, y_data, x_train, y_train, x_val, y_val = create_data()

    plt.figure(figsize=(16, 8))
    plt.plot(x_data, y_data, x_train, y_train, '.', x_val, y_val, 'o', markersize=4)
    plt.show()

    num_members = 5
    dataset_train, dataset_val = make_dataset(x_train, y_train, x_val, y_val, num_members)
    cfg = get_config()

    ensemble = models.Ensemble(num_members, 1, 2, device, cfg)

    train(ensemble, dataset_train, dataset_val)

    visualize_results(ensemble, x_data, y_data, x_train, y_train)

if __name__ == "__main__":
    main()
