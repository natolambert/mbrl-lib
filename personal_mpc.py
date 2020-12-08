import hydra
import numpy as np

# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import RcParams
import omegaconf
import torch

# import pytorch_sac
import mbrl.util as util
import gym
import mbrl.models as models
import time
import functools
import os
import logging

log = logging.getLogger(__name__)


def evaluate_action_sequences_(
    env, current_obs: np.ndarray, action_sequences: torch.Tensor, true_model=False
) -> torch.Tensor:
    all_rewards = torch.zeros(len(action_sequences))
    for i, sequence in enumerate(action_sequences):
        _, rewards, _ = util.rollout_env(env, current_obs, None, -1, plan=sequence)
        all_rewards[i] = rewards.sum().item()
    return all_rewards


PETS_LOG_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("rollout_length", "RL", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_score", "MSCORE", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
]

EVAL_LOG_FORMAT = [
    ("trial", "T", "int"),
    ("episode_reward", "R", "float"),
]


@hydra.main(config_path="conf", config_name="personal_mpc")  # personal_
def run(cfg: omegaconf.DictConfig):
    log.info(omegaconf.OmegaConf.to_yaml(cfg))
    env, term_fn, reward_fn = util.make_env(cfg)
    env.seed(cfg.seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.trial_length)
    env._elapsed_steps = 0
    env.reset()

    # # HC setup
    # pets_logger = pytorch_sac.Logger(
    #     os.getcwd(),
    #     save_tb=False,
    #     log_frequency=None,
    #     agent="pets",
    #     train_format=PETS_LOG_FORMAT,
    #     eval_format=EVAL_LOG_FORMAT,
    # )
    # obs_shape = env.observation_space.shape
    # act_shape = env.action_space.shape
    # from typing import cast, List
    # import mbrl.replay_buffer as replay_buffer
    # import mbrl

    # env_dataset_train, env_dataset_val = mbrl.util.create_ensemble_buffers(
    #     cfg, obs_shape, act_shape
    # )
    # env_dataset_train = cast(replay_buffer.BootstrapReplayBuffer, env_dataset_train)
    # env_dataset_train.load("/home/hiro/mbrl/exp/ref/halfcheetah/pets_replay_buffer_train.npz")
    # env_dataset_val.load("/home/hiro/mbrl/exp/ref/halfcheetah/pets_replay_buffer_val.npz")
    # dynamics_model = mbrl.util.create_dynamics_model(cfg, obs_shape, act_shape)
    # # name_obs_process_fn = "mbrl.env.pets_halfcheetah.HalfCheetahEnv.preprocess_fn"
    # #cfg.get("obs_process_fn", None)
    # # name_obs_process_fn = cfg.get("obs_process_fn", None)
    # # if name_obs_process_fn:
    # #     obs_process_fn = hydra.utils.get_method(cfg.obs_process_fn)
    # # else:
    # #     obs_process_fn = None
    # # dynamics_model = mbrl.models.DynamicsModelWrapper(
    # #     dynamics_model.model,
    # #     target_is_delta=cfg.target_is_delta,
    # #     normalize=cfg.normalize,
    # #     learned_rewards=cfg.learned_rewards,
    # #     obs_process_fn=obs_process_fn,
    # #     no_delta_list=cfg.get("no_delta_list", None),
    # # )
    # model_trainer = models.EnsembleTrainer(
    #     dynamics_model,
    #     env_dataset_train,
    #     dataset_val=env_dataset_val,
    #     logger=pets_logger,
    #     log_frequency=cfg.log_frequency_model,
    # )
    #
    # model_trainer.train(
    #     num_epochs=cfg.get("num_epochs_train_dyn_model", None),
    #     patience=cfg.patience,
    # )
    # dynamics_model.save(os.getcwd())
    # log.info("saved model")
    # exit()

    if "cheetah" in cfg.env:
        m_path = cfg.model_path_hc
    else:
        m_path = cfg.model_path
    model_cfg = util.get_hydra_cfg(m_path)
    dynamics_model = util.create_dynamics_model(
        model_cfg,
        env.observation_space.shape,
        env.action_space.shape,
        model_dir=m_path,
    )
    data = np.load(m_path + "replay_buffer_train.npz")
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

    if cfg.obs_gen == "set":
        obs = np.array(cfg.set_obs)
    elif cfg.obs_gen == "idx":
        obs = states[cfg.obs_idx]
    else:
        obs = env.reset()

    # obs = env.reset()
    # planner.reset()
    # actions_to_use: List[np.ndarray] = []
    # done = False
    # total_reward = 0
    # steps_trial = 0
    # while not done:
    #     # ------------- Planning using the learned model ---------------
    #     plan_time = 0.0
    #     if not actions_to_use:  # re-plan is necessary
    #         trajectory_eval_fn = functools.partial(
    #             model_env.evaluate_action_sequences,
    #             initial_state=obs,
    #             num_particles=cfg.num_particles,
    #             propagation_method=cfg.propagation_method,
    #         )
    #         start_time = time.time()
    #         plan, _ = planner.plan(
    #             model_env.action_space.shape,
    #             cfg.planning_horizon,
    #             trajectory_eval_fn,
    #         )
    #         plan_time = time.time() - start_time
    #
    #         actions_to_use.extend([a for a in plan[: cfg.replan_freq]])
    #     action = actions_to_use.pop(0)
    #
    #     # --- Doing env step and adding to model dataset ---
    #     next_obs, reward, done, _ = env.step(action)
    #     obs = next_obs
    #     total_reward += reward
    #     steps_trial += 1
    #     if steps_trial == cfg.trial_length:
    #         break
    #
    #     if True:
    #         print(f"Step {steps_trial}: Reward {reward:.3f}. Time: {plan_time:.3f}")
    #
    # pets_logger.log("eval/trial", 0, steps_trial)
    # pets_logger.log("eval/episode_reward", total_reward, steps_trial)
    #
    # # max_total_reward = max(max_total_reward, total_reward)
    # log.info(total_reward)

    if cfg.load_actions:
        # state = torch.load(cfg.actions_path + "state.pth")
        actions = torch.load(cfg.actions_path + "actions.pth")
        plans = torch.load(cfg.actions_path + "plans.pth")
        values = torch.load(cfg.actions_path + "values.pth")
    else:

        def trajectory_eval_fn(action_sequences):
            return evaluate_action_sequences_(
                use_env, obs, action_sequences, true_model=cfg.mpc_true_model
            )

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
    log.info(f"evaluating state {state}")

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

    log.info(f"mean action {np.mean(np.stack(actions))}")
    log.info(f"std action {np.std(np.stack(actions))}")
    log.info(f"mean value {np.mean(np.stack(values))}")
    log.info(f"std value {np.std(np.stack(values))}")

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
            "font.size": 14,
            # 'text.usetex': True,
        }
    )

    plt.style.use(latex_style_times)
    if not dim:
        d_a = np.shape(plans)[2]
        if d_a == 1:
            # 1 dimensional
            dims = [0]
        else:
            dims = np.arange(0, d_a)

    for d in dims:
        # plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()

        if bounds is not None:
            ax.set_ylim(bounds)

        len_dat = np.shape(plans)[1]
        t = np.arange(0, len_dat, 1)
        # the histogram of the data

        for pl in plans:
            plt.plot(t, pl[:, d], color="#7C65D7", alpha=0.7)
        # add a 'best fit' line
        # y = mlab.normpdf(bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=1)

        plt.ylabel("Action")
        plt.xlabel("Timestep")
        # if obs is not None:
        #     plt.title(f"Obs: {obs}")
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(False)
        # plt.text(0.3, 1.05, "obs:" + str(np.round(obs, 4)), size="small", transform=ax.transAxes)
        plt.text(0.3, 1.0, "dim:" + str(d), size="small", transform=ax.transAxes)
        plt.text(0.3, 0.95, os.getcwd()[-16:], size="small", transform=ax.transAxes)

        plt.savefig(f"plot_plans_d{d}.pdf")
        # maybe use evaluate action sequence on model


def visualize_actions(actions, dims=None, values=None, bounds=[-1, 1], obs=None):
    # Either 1 or 2 d for now
    plt.tight_layout()
    if not dims:
        d_a = np.shape(actions)[1]
        if d_a == 1:
            # 1 dimensional
            dims = [0]
        else:
            dims = np.arange(0, d_a)

    if values is not None:
        plt.rcParams["font.family"] = "Times"
        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # the histogram of the data
        weights = np.ones_like(values) / float(len(values))
        n, bins, patches = plt.hist(
            values, bins=25, facecolor="#D76565", alpha=0.75, weights=weights
        )

        # add a 'best fit' line
        # y = mlab.normpdf(bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=1)

        plt.xlabel("Values")
        plt.ylabel("Probability")
        # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
        # plt.axis([40, 160, 0, 0.03])
        plt.grid(False)

        # add obs, mean, var for logging easily
        # plt.text(0.1, 1, "obs:" + str(np.round(obs,4)), size="small", transform=ax.transAxes)
        plt.text(
            0.6,
            1,
            "mean: " + str(np.round(np.mean(values), 4)),
            size="small",
            transform=ax.transAxes,
        )
        plt.text(
            0.6,
            0.95,
            "stdev: " + str(np.round(np.std(values), 4)),
            size="small",
            transform=ax.transAxes,
        )
        plt.text(0.1, 0.95, os.getcwd()[-16:], size="small", transform=ax.transAxes)

        plt.savefig("plot_values.pdf")

    for d in dims:
        # number line of actions
        plt.rcParams["font.family"] = "Times"
        fig, ax = plt.subplots()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # the histogram of the data
        weights = np.ones_like(values) / float(len(values))
        n, bins, patches = plt.hist(
            actions[:, d], bins=25, facecolor="green", alpha=0.75, weights=weights
        )
        # plt.text(0.1, 1, "obs:" + str(np.round(obs, 4)), size="small", transform=ax.transAxes)
        plt.text(0.1, 1, "dim:" + str(d), size="small", transform=ax.transAxes)
        plt.text(
            0.6,
            1,
            "mean: " + str(np.round(np.mean(actions[:, d]), 4)),
            size="small",
            transform=ax.transAxes,
        )
        plt.text(
            0.6,
            0.95,
            "stdev: " + str(np.round(np.std(actions[:, d]), 4)),
            size="small",
            transform=ax.transAxes,
        )
        plt.text(0.1, 0.95, os.getcwd()[-16:], size="small", transform=ax.transAxes)

        # add a 'best fit' line
        # y = mlab.normpdf(bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=1)
        if bounds is not None:
            ax.set_xlim(bounds)

        plt.xlabel("Actions")
        plt.ylabel("Probability")
        # if obs is not None:
        #     plt.title(f"Obs: {obs}")

        # plt.axis([40, 160, 0, 0.03])
        plt.grid(False)

        plt.savefig(f"plot_action_d{d}.pdf")
        # return

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
