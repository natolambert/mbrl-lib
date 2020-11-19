import hydra
import numpy as np
import omegaconf
import torch

import mbrl.util as util
import gym
import mbrl.models as models
import time


def evaluate_action_sequences_(
    env, current_obs: np.ndarray, action_sequences: torch.Tensor, true_model=False
) -> torch.Tensor:
    all_rewards = torch.zeros(len(action_sequences))
    for i, sequence in enumerate(action_sequences):
        if not true_model:
            # model creates tensors
            _, rewards, _ = util.rollout_model_env(
                env, current_obs, plan=sequence.cpu().numpy()
            )
            # rewards = rewards.cpu().numpy()
        else:
            # mujoco creates np
            _, rewards, _ = util.rollout_env(env, current_obs, None, -1, plan=sequence)
        all_rewards[i] = rewards.sum().item()
    return all_rewards


@hydra.main(config_path="conf", config_name="personal_mpc")  # personal_
def run(cfg: omegaconf.DictConfig):
    # cfg = omegaconf.OmegaConf.create(
    #     {"env": "gym___HalfCheetah-v2", "term_fn": "no_termination"}
    # )
    env, term_fn, reward_fn = util.make_env(cfg)
    env.seed(cfg.seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    # controller = mbrl.planning.CEMPlanner(
    #     5,
    #     0.1,
    #     500,
    #     env.action_space.low,
    #     env.action_space.high,
    #     0.1,
    #     torch.device("cuda:0"),
    # )
    # TODO add config for real env
    # if cfg.real_env:
    #     a = 1
    #
    # else:
    model_cfg = util.get_hydra_cfg(cfg.model_path)
    dynamics_model = util.create_dynamics_model(
        model_cfg,
        env.observation_space.shape,
        env.action_space.shape,
        model_dir=cfg.model_path,
    )

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

    step = 0
    obs = env.reset()

    actions, plans, values = repeat_mpc(
        use_env, obs, planner, 10, cfg, trajectory_eval_fn
    )

    # on real env
    # actions, plans, values = repeat_mpc(model_env, obs, planner, 10, cfg, trajectory_eval_fn)

    done = False
    total_reward = 0.0
    while not done:
        plan, plan_value = planner.plan(env.action_space.shape, 30, trajectory_eval_fn)

        next_obs, reward, done, _ = env.step(plan[0])
        total_reward += reward
        obs = next_obs
        step += 1
        print(step, reward)

    print("total reward: ", total_reward)


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
        action_sequence, pred_val = controller.plan(
            env.action_space.shape, cfg.planning_horizon, plan_eval_fn
        )
        actions.append(action_sequence[0])
        plans.append(action_sequence)
        values.append(pred_val)
        controller.reset()
        plan_time = time.time() - start_time
        if (n + 1) % 10 == 0:
            print(f"eval action {n+1}, avg time {plan_time/10}")

    print(f"mean action {np.mean(np.stack(actions))}")
    print(f"std action {np.std(np.stack(actions))}")
    print(f"mean value {np.mean(np.stack(values))}")
    print(f"std value {np.std(np.stack(values))}")

    return np.stack(actions), np.stack(plans), np.stack(values)


def visualize_plans(plans):
    # TODO get the simulated states from the controller as well

    # maybe use evaluate action sequence on model
    return 0


def visualize_actions(actions, dims=None):
    # Either 1 or 2 d for now
    if not dims:
        d_a = np.shape(actions)[1]
        if d_a == 1:
            # 1 dimensional
            dims = [0]
        else:
            dims = [0, 1]

    if len(dims) == 1:
        # number line of actions
        return 0

    if len(dims) == 2:
        # 2d plot of actions
        return 0


if __name__ == "__main__":
    run()
