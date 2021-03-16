import hydra
import numpy as np

# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import RcParams
import omegaconf
import torch

# import pytorch_sac
import mbrl.util.common as common_util
import mbrl.util.mujoco as mujoco_util
import mbrl
from mbrl.planning import ImitativeAgent

import gym
import mbrl.models as models
import time
import functools
import os
from typing import List, cast
import logging

log = logging.getLogger(__name__)


def evaluate_action_sequences_(
    env, current_obs: np.ndarray, action_sequences: torch.Tensor, true_model=False
) -> torch.Tensor:
    all_rewards = torch.zeros(len(action_sequences))
    for i, sequence in enumerate(action_sequences):
        # _, rewards, _ = util.rollout_env(env, current_obs, None, -1, plan=sequence)
        _, rewards, _ = mujoco_util.rollout_mujoco_env(
            env, current_obs, None, -1, plan=sequence
        )
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


@hydra.main(config_path="conf", config_name="mpc")  # personal_
def run(cfg: omegaconf.DictConfig):
    log.info(omegaconf.OmegaConf.to_yaml(cfg))
    env, term_fn, reward_fn = mujoco_util.make_env(cfg)
    env.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.overrides.trial_length)
    env._elapsed_steps = 0
    env.reset()
    rng = np.random.default_rng(seed=cfg.seed)
    debug_mode = cfg.debug_mode
    mp = cfg.model_id
    # for mp in range(len(cfg.model_paths)):
    m_path = cfg.model_paths[mp]
    log.info(f"evaluating policy from logdir {m_path}, n {mp} of max {len(cfg.model_paths)}")

    #  POLICY TRAINING cfg is not ensemble, weirdly hard to set this
    if "cheetah" in cfg.overrides.env:
        m_path = cfg.model_path_hc

    model_cfg = common_util.load_hydra_cfg(m_path)
    dynamics_model = common_util.create_dynamics_model(
        model_cfg,
        env.observation_space.shape,
        env.action_space.shape,
        model_dir=m_path,
    )

    work_dir = os.getcwd()
    logger = mbrl.logger.Logger(work_dir)
    logger.register_group("MPC_earlywork", EVAL_LOG_FORMAT, color="green")

    # cfg.overrides.model_batch_size = 1
    # FOR TRAINNG POLICY, train_is_bootstrap is False (change for models, if want ensemble)
    data, data_val = common_util.create_replay_buffers(
        cfg, env.observation_space.shape, env.action_space.shape, m_path, train_is_bootstrap=False,
    )
    data = cast(mbrl.replay_buffer.BootstrapReplayBuffer, data)
    data_val = cast(mbrl.replay_buffer.BootstrapReplayBuffer, data_val)
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, term_fn, reward_fn, seed=cfg.seed
    )


    REWARDS = []
    if cfg.mode == "recompute":

        # model_trainer = mbrl.models.DynamicsModelTrainer(
        #     dynamics_model, dataset_train, dataset_val=dataset_val, logger=logger
        # )

        agent = mbrl.planning.create_trajectory_optim_agent_for_model(
            model_env,
            cfg.algorithm.agent,
            num_particles=cfg.algorithm.num_particles,
            propagation_method=cfg.algorithm.propagation_method,
        )
        a_old = []
        a_new = []
        rs = []
        data_new, data_new_val = common_util.create_replay_buffers(
            cfg, env.observation_space.shape, env.action_space.shape, train_is_bootstrap=False
        )
        data_new.num_members = 1
        data_new = cast(mbrl.replay_buffer.BootstrapReplayBuffer, data_new)
        data_new_val.num_members = 1
        data_new_val = cast(mbrl.replay_buffer.BootstrapReplayBuffer, data_new_val)

        print(f"... Training datset of length {len(data)}")
        for n, batch in enumerate(data):
            if n % 50 == 0:
                print(f"... recomuption on datapoint {n+1}")
            if len(batch) == 1:
                batch = batch[0]
            obs, action_old, next_obs, reward, done = batch
            # NOTE, the training data does NOT contain all of the actions in order
            if done:
                agent.reset(planning_horizon=20)

            action = agent.act(obs)
            a_old.append(action_old)
            a_new.append(action)
            rs.append(reward)
            data_new.add(obs, action, next_obs, reward, done)
            # trajectory_eval_fn = functools.partial(
            #     env.evaluate_action_sequences,
            #     initial_state=obs,
            #     num_particles=cfg.num_particles,
            #     propagation_method=cfg.propagation_method,
            # )
            # action_sequence, pred_val = controller.plan(
            #     env.action_space.shape,
            #     cfg.planning_horizon,
            #     trajectory_eval_fn,
            # )

        for batch in data_val:
            if len(batch) == 1:
                batch = batch[0]
            obs, action_old, next_obs, reward, done = batch
            # NOTE, the training data does NOT contain all of the actions in order
            if done:
                agent.reset(planning_horizon=20)

            action = agent.act(obs)
            data_new_val.add(obs, action, next_obs, reward, done)
            # trajectory_eval_fn = functools.partial(
            #     env.evaluate_action_sequences,
            #     initial_state=obs,
            #     num_particles=cfg.num_particles,
            #     propagation_method=cfg.propagation_method,
            # )
            # action_sequence, pred_val = controller.plan(
            #     env.action_space.shape,
            #     cfg.planning_horizon,
            #     trajectory_eval_fn,
            # )

        a_old_n = np.array(a_old)
        a_new_n = np.array(a_new)
        data_new.save(m_path + "_recompute")
        np.savez(m_path + "action_old.npz", a_old_n)
        np.savez(m_path + "action_new.npz", a_new_n)

        if False:
            visualize_actions(a_old_n, values=np.stack(rs), st="old")
            visualize_actions(a_new_n, values=np.stack(rs), st="new")

        if True:
            policy, policy_trainer = create_policy(
                cfg, env, data_new, data_new_val, logger
            )

            # train imitative policy
            common_util.train_policy_and_save_model_and_data(
                policy, policy_trainer, cfg, data_new, data_new_val, os.getcwd(), callback=eval_pol_callback
            )
            agent_reactive = ImitativeAgent(policy)
            rews = run_trials_pol(cfg, agent_reactive, env, policy, data, data_val, rng)

    elif cfg.mode == "imitate":


        # # load data
        # data, data_val = common_util.create_replay_buffers(
        #     cfg, env.observation_space.shape, env.action_space.shape, m_path
        # )
        # data.num_members = 1
        # data_val.num_members = 1
        # data = cast(mbrl.replay_buffer.BootstrapReplayBuffer, data)
        # data_val = cast(mbrl.replay_buffer.BootstrapReplayBuffer, data_val)
        # data.member_indices = List[List[int]] = [None for _ in range(1)]
        # data_val.member_indices = List[List[int]] = [None for _ in range(1)]

        # data = np.load(m_path + "replay_buffer_train.npz")
        # data_val = np.load(m_path + "replay_buffer_val.npz")
        data_orig = data
        data_val_orig = data_val

        cfg.dynamics_model.model.ensemble_size = 1
        policy, policy_trainer = create_policy(cfg, env, data, data_val, logger)

        # look at per-epoch training
        if cfg.per_epoch:
            epochs = cfg.overrides.num_epochs_train_model
            cfg.overrides.num_epochs_train_model = 1
            for e in range(epochs):
                log.info(f"Epoch {e}/{epochs}")
                # train imitative policy
                common_util.train_policy_and_save_model_and_data(
                    policy, policy_trainer, cfg, data_orig, data_val_orig, os.getcwd(),
                )

                agent_reactive = ImitativeAgent(policy)
                rews = run_trials_pol(cfg, agent_reactive, env, policy, data, data_val, rng, log=False)
                REWARDS.append(rews)
                logger.log_data(
                    "MPC_earlywork", {"trial": e, "episode_reward": rews.squeeze()}
                )

            cfg.overrides.num_epochs_train_model = epochs

        else:
            # train imitative policy
            common_util.train_policy_and_save_model_and_data(
                policy, policy_trainer, cfg, data, data_val, os.getcwd()
            )

            agent_reactive = ImitativeAgent(policy)
            rews = run_trials_pol(cfg, agent_reactive, env, policy, data, data_val, rng)

    elif cfg.mode == "mpc":

        # TODO add state to the plot of the desired actions distribution
        model_env = models.ModelEnv(
            env, dynamics_model, term_fn, reward_fn, seed=cfg.seed
        )
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

        # Old code below
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


def run_trials_pol(cfg, agent, env, policy, data_train, data_val, rng, log=True):
    rs = []
    for i in range(cfg.num_eval):
        agent.reset()

        obs = env.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0
        env_steps = 0
        cum_reward = 0
        while not done:
            if log:
                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = mbrl.util.step_env_and_populate_dataset(
                    env,
                    obs.astype(np.float32),
                    agent,
                    {},
                    data_train,
                    data_val,
                    cfg.algorithm.increase_val_set,
                    cfg.overrides.validation_ratio,
                    rng,
                    policy.update_normalizer,
                )
            else:
                action = agent.act(obs.astype(np.float32))
                next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            cum_reward += reward
            if steps_trial == cfg.overrides.trial_length:
                break

            if cfg.debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")
        # print(f"Cumulative reward {cum_reward}")
        rs.append(cum_reward)
    if cfg.debug_mode:
        print(rs)
        print(f"Summary: mean {np.mean(rs)}, sdt {np.std(rs)}")
    return np.stack(rs)

def eval_pol_callback(train_iteration, epoch,total_avg_loss, train_score, eval_score, best_val_score, env, cfg):
    print(f"running callback at epoch {train_iteration}")
    for itr in cfg.n_eval:
        print(itr)

def create_policy(cfg, env, data_train, data_val, logger):
    # creates imitative polict object
    policy = common_util.create_policy(
        cfg,
        env.observation_space.shape,
        env.action_space.shape,
    )

    # policy trainer like model trainer
    policy_trainer = mbrl.models.PolicyTrainer(
        policy, data_train, dataset_val=data_val, logger=logger
    )

    return policy, policy_trainer


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


def visualize_actions(
    actions, dims=None, values=None, bounds=[-1, 1], obs=None, st=None
):
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

        plt.savefig(f"plot_action_d{d}{st}.pdf")
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
