# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List, Optional, cast

import gym
import numpy as np
import omegaconf
import torch

import mbrl.logger
import mbrl.math
import mbrl.models
import mbrl.planning
import mbrl.replay_buffer
import mbrl.types
import mbrl.util

EVAL_LOG_FORMAT = [
    ("trial", "T", "int"),
    ("episode_reward", "R", "float"),
]


def get_rollout_schedule(cfg: omegaconf.DictConfig) -> List[int]:
    max_horizon = cfg.overrides.get(
        "max_planning_horizon", cfg.algorithm.agent.planning_horizon
    )
    if "trial_reach_max_horizon" in cfg.overrides:
        return [1, cfg.overrides.trial_reach_max_horizon, 1, max_horizon]
    else:
        return [1, cfg.overrides.num_trials, max_horizon, max_horizon]


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.logger.Logger(work_dir)
        logger.register_group("pets_eval", EVAL_LOG_FORMAT, color="green")

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.create_proprioceptive_model(cfg, obs_shape, act_shape)

    dataset_train, dataset_val = mbrl.util.create_replay_buffers(
        cfg,
        obs_shape,
        act_shape,
        train_is_weighted=cfg.algorithm.weighted,
        train_is_bootstrap=isinstance(dynamics_model.model, mbrl.models.Ensemble),
        collect_trajectories=cfg.algorithm.log_trajs,
        rng=rng,
    )

    if cfg.algorithm.weighted:
        from mbrl.replay_buffer import update_weights
        dataset_train = cast(
            mbrl.replay_buffer.WeightedBootstrapReplayBuffer, dataset_train
        )
        dataset_train.setup()
        n_initial = 5
    else:
        dataset_train = cast(mbrl.replay_buffer.BootstrapReplayBuffer, dataset_train)
        n_initial = cfg.algorithm.initial_exploration_steps

    # when doing traj, # of transitions for random doesn't work
    mbrl.util.rollout_agent_trajectories(
        env,
        n_initial,  # ,
        mbrl.planning.RandomAgent(env),
        {},
        rng,
        trial_length=cfg.overrides.trial_length,
        train_dataset=dataset_train,
        val_dataset=dataset_val,
        val_ratio=cfg.overrides.validation_ratio,
        callback=dynamics_model.update_normalizer,
        collect_full_trajectories=cfg.algorithm.log_trajs,
        store_weights=cfg.algorithm.weighted,
    )
    mbrl.util.save_buffers(dataset_train, dataset_val, work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.DynamicsModelTrainer(
        dynamics_model,
        dataset_train,
        dataset_val=dataset_val,
        optim_lr=cfg.overrides.model_lr,
        weight_loss=cfg.algorithm.weighted,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    for trial in range(cfg.overrides.num_trials):
        obs = env.reset()

        planning_horizon = int(
            mbrl.math.truncated_linear(*(get_rollout_schedule(cfg) + [trial + 1]))
        )

        agent.reset(planning_horizon=planning_horizon)

        # DIFFERENT STRUCTURE BECAUSE WEIGHTED USES TRAJ ONLY
        if cfg.algorithm.weighted or cfg.algorithm.log_trajs:
            total_reward = mbrl.util.rollout_agent_trajectories(
                env,
                1,
                agent,
                {},
                rng,
                trial_length=cfg.overrides.trial_length,
                train_dataset=dataset_train,
                val_dataset=dataset_val,
                val_ratio=cfg.overrides.validation_ratio,
                callback=dynamics_model.update_normalizer,
                collect_full_trajectories=cfg.algorithm.log_trajs,
                store_weights=cfg.algorithm.weighted,
            )[0]

            if cfg.algorithm.weighted: update_weights(dataset_train)
            mbrl.util.train_model_and_save_model_and_data(
                dynamics_model,
                model_trainer,
                cfg,
                dataset_train,
                dataset_val,
                work_dir,
            )

        else:
            done = False
            total_reward = 0.0
            steps_trial = 0
            while not done:
                # --------------- Model Training -----------------
                if steps_trial == 0 or env_steps % cfg.algorithm.freq_train_model == 0:
                    mbrl.util.train_model_and_save_model_and_data(
                        dynamics_model,
                        model_trainer,
                        cfg,
                        dataset_train,
                        dataset_val,
                        work_dir,
                    )

                # --- Doing env step using the agent and adding to model dataset ---
                next_obs, reward, done, _ = mbrl.util.step_env_and_populate_dataset(
                    env,
                    obs,
                    agent,
                    {},
                    dataset_train,
                    dataset_val,
                    cfg.algorithm.increase_val_set,
                    cfg.overrides.validation_ratio,
                    rng,
                    dynamics_model.update_normalizer,
                )

                obs = next_obs
                total_reward += reward
                steps_trial += 1
                env_steps += 1
                if steps_trial == cfg.overrides.trial_length:
                    break

                if debug_mode:
                    print(f"Step {env_steps}: Reward {reward:.3f}.")

        if logger is not None:
            logger.log_data(
                "pets_eval", {"trial": current_trial, "episode_reward": total_reward}
            )
        current_trial += 1
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
