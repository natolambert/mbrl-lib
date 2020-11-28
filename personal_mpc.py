import hydra
import numpy as np

# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import RcParams

import omegaconf
import torch

import mbrl.util as util
import gym
import mbrl.models as models
import time
import functools
import os


def evaluate_action_sequences_(
    env, current_obs: np.ndarray, action_sequences: torch.Tensor, true_model=False
) -> torch.Tensor:
    all_rewards = torch.zeros(len(action_sequences))
    for i, sequence in enumerate(action_sequences):
        _, rewards, _ = util.rollout_env(env, current_obs, None, -1, plan=sequence)
        all_rewards[i] = rewards.sum().item()
    return all_rewards


@hydra.main(config_path="conf", config_name="personal_mpc")  # personal_
def run(cfg: omegaconf.DictConfig):
    print(omegaconf.OmegaConf.to_yaml(cfg))
    env, term_fn, reward_fn = util.make_env(cfg)
    env.seed(cfg.seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    env._elapsed_steps = 0

    model_cfg = util.get_hydra_cfg(cfg.model_path)
    dynamics_model = util.create_dynamics_model(
        model_cfg,
        env.observation_space.shape,
        env.action_space.shape,
        model_dir=cfg.model_path,
    )
    data = np.load(cfg.model_path + "/replay_buffer_train.npz")
    states = data["obs"]
    # acts = data["action"]
    # TODO add state to the plot of the desired actions distribution
    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, seed=cfg.seed)
    if cfg.mpc_true_model:
        use_env = env
    else:
        use_env = model_env

    # TODO config-ize the model env
    cfg.planner.action_lb = env.action_space.low.tolist()
    cfg.planner.action_ub = env.action_space.high.tolist()
    planner = hydra.utils.instantiate(cfg.planner)

    def trajectory_eval_fn(action_sequences):
        return evaluate_action_sequences_(
            use_env, obs, action_sequences, true_model=cfg.mpc_true_model
        )

    if cfg.obs_gen == "set":
        obs = np.array(cfg.set_obs)
    elif cfg.obs_gen == "idx":
        obs = states[cfg.obs_idx]
    else:
        obs = env.reset()

    if cfg.load_actions:
        # state = torch.load(cfg.actions_path + "state.pth")
        actions = torch.load(cfg.actions_path + "actions.pth")
        plans = torch.load(cfg.actions_path + "plans.pth")
        values = torch.load(cfg.actions_path + "values.pth")
    else:
        actions, plans, values = repeat_mpc(
            use_env, obs, planner, cfg.mpc_repeat, cfg, trajectory_eval_fn
        )
    # on real env
    # actions, plans, values = repeat_mpc(model_env, obs, planner, 10, cfg, trajectory_eval_fn)

    visualize_plans(plans, obs=obs)
    visualize_actions(actions, values=values, obs=obs)
    #
    # quit()
    # done = False
    # total_reward = 0.0
    # while not done:
    #     plan, plan_value = planner.plan(env.action_space.shape, 30, trajectory_eval_fn)
    #
    #     next_obs, reward, done, _ = env.step(plan[0])
    #     total_reward += reward
    #     obs = next_obs
    #     step += 1
    #     print(step, reward)
    #
    # print("total reward: ", total_reward)


def repeat_mpc(env, state, controller, repeat, cfg, trajectory_eval_fn):
    plans = []
    actions = []
    values = []

    # start_time = time.time()

    def plan_eval_fn(action_sequences):
        return evaluate_action_sequences_(
            env, state, action_sequences, true_model=cfg.mpc_true_model
        )

    for n in range(repeat):
        start_time = time.time()

        if cfg.mpc_true_model:
            action_sequence, pred_val = controller.plan(
                env.action_space.shape, cfg.planning_horizon, plan_eval_fn
            )
        else:
            trajectory_eval_fn = functools.partial(
                env.evaluate_action_sequences,
                initial_state=state,
                num_particles=cfg.num_particles,
                propagation_method=cfg.propagation_method,
            )
            action_sequence, pred_val = controller.plan(
                env.action_space.shape,
                cfg.planning_horizon,
                trajectory_eval_fn,
            )
        actions.append(action_sequence[0])
        plans.append(action_sequence)
        values.append(pred_val)
        controller.reset()
        plan_time = time.time() - start_time
        if (n + 1) % 1 == 0:
            print(f"eval action {n + 1}, avg time {plan_time / 10}")

    dir = os.getcwd()
    torch.save(state, dir + "/state.pth")
    torch.save(np.stack(actions), dir + "/actions.pth")
    torch.save(np.stack(plans), dir + "/plans.pth")
    torch.save(np.stack(values), dir + "/values.pth")
    if not cfg.mpc_true_model:
        env.dynamics_model.save(dir)

    print(f"mean action {np.mean(np.stack(actions))}")
    print(f"std action {np.std(np.stack(actions))}")
    print(f"mean value {np.mean(np.stack(values))}")
    print(f"std value {np.std(np.stack(values))}")

    return np.stack(actions), np.stack(plans), np.stack(values)


def visualize_plans(plans, dim=None, values=None, bounds=[-1, 1], obs=None):
    # TODO get the simulated states from the controller as well
    # Only works for one set of actions
    # number line of actions
    import matplotlib.font_manager

    matplotlib.font_manager.findfont("serif", rebuild_if_missing=False)
    matplotlib.font_manager.findfont("serif", fontext="afm", rebuild_if_missing=False)

    plt.tight_layout()
    latex_style_times = RcParams(
        {
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 18,
            # 'text.usetex': True,
        }
    )

    plt.style.use(latex_style_times)

    # plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if bounds is not None:
        ax.set_ylim(bounds)

    len_dat = np.shape(plans)[1]
    t = np.arange(0, len_dat, 1)
    # the histogram of the data

    for pl in plans:
        plt.plot(t, pl, color="#7C65D7", alpha=0.7)
    # add a 'best fit' line
    # y = mlab.normpdf(bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.ylabel("Action")
    plt.xlabel("Timestep")
    if obs is not None:
        plt.title(f"Obs: {obs}")
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(False)

    plt.savefig("plans.pdf")
    # maybe use evaluate action sequence on model
    return 0


def visualize_actions(actions, dims=None, values=None, bounds=[-1, 1], obs=None):
    # Either 1 or 2 d for now
    plt.tight_layout()
    if not dims:
        d_a = np.shape(actions)[1]
        if d_a == 1:
            # 1 dimensional
            dims = [0]
        else:
            dims = [0, 1]

    if values is not None:
        plt.rcParams["font.family"] = "Times"
        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # the histogram of the data
        n, bins, patches = plt.hist(actions, 25, facecolor="#D76565", alpha=0.75)

        # add a 'best fit' line
        # y = mlab.normpdf(bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=1)

        plt.xlabel("Values")
        plt.ylabel("Count")
        # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(False)

        plt.savefig("values.pdf")

    if len(dims) == 1:
        # number line of actions
        plt.rcParams["font.family"] = "Times"
        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # the histogram of the data
        n, bins, patches = plt.hist(actions, 25, facecolor="green", alpha=0.75)

        # add a 'best fit' line
        # y = mlab.normpdf(bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=1)
        if bounds is not None:
            ax.set_xlim(bounds)

        plt.xlabel("Actions")
        plt.ylabel("Count")
        if obs is not None:
            plt.title(f"Obs: {obs}")
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(False)

        plt.savefig("action_1d.pdf")
        return

    # if len(dims) == 2:
    #     # 2d plot of actions
    #     # Libraries
    #     raise NotImplementedError("TODO 2d Plot")
    #
    #     # Create data: 200 points
    #     data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    #     x, y = data.T
    #
    #     if bounds is not None:
    #         ax.set_xlim(bounds)
    #         ax.set_ylim(bounds)
    #     # Create a figure with 6 plot areas
    #     fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
    #
    #     # Everything sarts with a Scatterplot
    #     axes[0].set_title("Scatterplot")
    #     axes[0].plot(x, y, "ko")
    #     # As you can see there is a lot of overplottin here!
    #
    #     # Thus we can cut the plotting window in several hexbins
    #     nbins = 20
    #     axes[1].set_title("Hexbin")
    #     axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)
    #
    #     # 2D Histogram
    #     axes[2].set_title("2D Histogram")
    #     axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
    #
    #     # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    #     k = kde.gaussian_kde(data.T)
    #     xi, yi = np.mgrid[
    #         x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
    #     ]
    #     zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #
    #     # plot a density
    #     axes[3].set_title("Calculate Gaussian KDE")
    #     axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
    #
    #     # add shading
    #     axes[4].set_title("2D Density with shading")
    #     axes[4].pcolormesh(
    #         xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap=plt.cm.BuGn_r
    #     )
    #
    #     # contour
    #     axes[5].set_title("Contour")
    #     axes[5].pcolormesh(
    #         xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap=plt.cm.BuGn_r
    #     )
    #     axes[5].contour(xi, yi, zi.reshape(xi.shape))
    #     plt.savefig("action_2d.pdf")
    #     if obs is not None:
    #         plt.title(f"Obs: {obs}")
    #
    #     return


if __name__ == "__main__":
    run()
