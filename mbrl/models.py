import abc
import itertools
import functools
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import pytorch_sac
import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

import mbrl.math
import mbrl.types

from . import replay_buffer


def truncated_normal_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        mbrl.math.truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)


# ------------------------------------------------------------------------ #
# Model classes
# ------------------------------------------------------------------------ #
class Model(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device(device)
        self.is_ensemble = False
        self.to(device)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def loss(
        self,
        model_in: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Returns the non-reduced score
        pass

    @abc.abstractmethod
    def save(self, path: str):
        pass

    @abc.abstractmethod
    def load(self, path: str):
        pass

    def update(
        self,
        model_in: Union[torch.Tensor, Sequence[torch.Tensor]],
        target: Union[torch.Tensor, Sequence[torch.Tensor]],
        optimizer: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
    ) -> float:
        assert not isinstance(optimizer, Sequence)
        optimizer = cast(torch.optim.Optimizer, optimizer)
        self.train()
        optimizer.zero_grad()
        loss = self.loss(model_in, target)
        loss.backward()
        optimizer.step(None)
        return loss.item()


class GaussianMLP(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        use_silu: bool = False,
        min_logvar_init = np.log(0.01 ** 2),
        max_logvar_init = np.log(1.28 ** 2),
    ):
        super(GaussianMLP, self).__init__(in_size, out_size, device)
        activation_cls = nn.SiLU if use_silu else nn.ReLU
        hidden_layers = [nn.Sequential(nn.Linear(in_size, hid_size), activation_cls())]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hid_size, hid_size), activation_cls())
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.mean_and_logvar = nn.Linear(hid_size, 2 * out_size)
        self.min_logvar = nn.Parameter(
            min_logvar_init * torch.ones(1, out_size, requires_grad=True)
        )
        self.max_logvar = nn.Parameter(
            max_logvar_init * torch.ones(1, out_size, requires_grad=True)
        )
        self.out_size = out_size

        self.apply(truncated_normal_init)
        self.to(self.device)

    def forward(self, x: torch.Tensor, **_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        mean = mean_and_logvar[:, : self.out_size]
        logvar = mean_and_logvar[:, self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean, pred_logvar = self.forward(model_in)
        nll = mbrl.math.gaussian_nll(pred_mean, pred_logvar, target)
        loss = nll + 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()

        # normalize the loss by output dimension (eaiser to interpret)
        loss = loss / self.out_size
        return loss

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in)
            return F.mse_loss(pred_mean, target, reduction="none").mean(dim=1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Ensemble(Model):
    def __init__(
        self,
        ensemble_size: int,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        member_cfg: omegaconf.DictConfig,
    ):
        super().__init__(in_size, out_size, device)
        self.is_ensemble = True
        self.members = []
        for i in range(ensemble_size):
            model = hydra.utils.instantiate(member_cfg)
            self.members.append(model)
        self.members = nn.ModuleList(self.members)
        self.to(device)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, item):
        return self.members[item]

    def __iter__(self):
        return iter(self.members)

    def _default_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = [model(x) for model in self.members]
        all_means = torch.stack([p[0] for p in predictions], dim=0)
        if predictions[0][1] is not None:
            all_logvars = torch.stack([p[1] for p in predictions], dim=0)
        else:
            all_logvars = None
        return all_means, all_logvars

    def _forward_from_indices(
        self, x: torch.Tensor, model_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        means = torch.empty((batch_size, self.out_size), device=self.device)
        logvars = torch.empty((batch_size, self.out_size), device=self.device)
        has_logvar = True
        for i, member in enumerate(self.members):
            model_idx = model_indices == i
            mean, logvar = member(x[model_idx])
            means[model_idx] = mean
            if logvar is not None:
                logvars[model_idx] = logvar
            else:
                has_logvar = False
        if not has_logvar:
            logvars = None
        return means, logvars

    def _forward_random_model(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        model_indices = torch.randint(
            len(self.members), size=(batch_size,), device=self.device
        )
        return self._forward_from_indices(x, model_indices)

    def _forward_expectation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_means, all_logvars = self._default_forward(x)
        mean = all_means.mean(dim=0)
        logvar = all_logvars.mean(dim=0) if all_logvars is not None else None
        return mean, logvar

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        propagation: Optional[str] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if propagation is None:
            return self._default_forward(x)
        if propagation == "random_model":
            return self._forward_random_model(x)
        if propagation == "fixed_model":
            assert (
                propagation_indices is not None
            ), "When using propagation='fixed_model', `propagation_indices` must be provided."
            return self._forward_from_indices(x, propagation_indices)
        if propagation == "expectation":
            return self._forward_expectation(x)
        raise ValueError(
            f"Invalid propagation method {propagation}. Valid options are: "
            f"'random_model', 'fixed_model', 'expectation'."
        )

    def loss(
        self,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        avg_ensemble_loss: torch.Tensor = 0.0
        for i, model in enumerate(self.members):
            model.train()
            loss = model.loss(inputs[i], targets[i])
            avg_ensemble_loss += loss
        return avg_ensemble_loss / len(self.members)

    def update(
        self,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        optimizers: Sequence[torch.optim.Optimizer],
    ) -> float:
        avg_ensemble_loss = 0
        for i, model in enumerate(self.members):
            model.train()
            optimizers[i].zero_grad()
            loss = model.loss(inputs[i], targets[i])
            # print(f"{i}-th model's loss = {loss}")
            loss.backward()
            optimizers[i].step(None)
            avg_ensemble_loss += loss.item()
        return avg_ensemble_loss / len(self.members)

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inputs = [model_in for _ in range(len(self.members))]
        targets = [target for _ in range(len(self.members))]

        with torch.no_grad():
            avg_ensemble_score = torch.tensor(0.0)
            for i, model in enumerate(self.members):
                model.eval()
                score = model.eval_score(inputs[i], targets[i])
                avg_ensemble_score = score + avg_ensemble_score
            return avg_ensemble_score / len(self.members)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class DynamicsModelWrapper:
    _MODEL_FNAME = "model.pth"

    def __init__(
        self,
        model: Model,
        target_is_delta: bool = True,
        normalize: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
    ):
        self.model = model
        self.normalizer: Optional[mbrl.math.Normalizer] = None
        if normalize:
            self.normalizer = mbrl.math.Normalizer(
                self.model.in_size, self.model.device
            )
        self.device = self.model.device
        self.learned_rewards = learned_rewards
        self.target_is_delta = target_is_delta
        self.no_delta_list = no_delta_list if no_delta_list else []
        self.obs_process_fn = obs_process_fn

    def update_normalizer(self, batch: mbrl.types.RLBatch):
        obs, action, next_obs, reward, _ = batch
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=1)
        if self.normalizer:
            self.normalizer.update_stats(model_in_np)

    def _get_model_input_from_np(
        self, obs: np.ndarray, action: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=1)
        if self.normalizer:
            # Normalizer lives on device
            return self.normalizer.normalize(model_in_np)
        return torch.from_numpy(model_in_np).to(device)

    def _get_model_input_from_tensors(self, obs: torch.Tensor, action: torch.Tensor):
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in = torch.cat([obs, action], axis=1)
        if self.normalizer:
            model_in = self.normalizer.normalize(model_in)
        return model_in

    def _get_model_input_and_target_from_batch(
        self, batch: mbrl.types.RLBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action, next_obs, reward, _ = batch
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[:, dim] = next_obs[:, dim]
        else:
            target_obs = next_obs

        model_in = self._get_model_input_from_np(obs, action, self.device)
        if self.learned_rewards:
            target = torch.from_numpy(
                np.concatenate([target_obs, np.expand_dims(reward, axis=1)], axis=1)
            ).to(self.device)
        else:
            target = torch.from_numpy(target_obs).to(self.device)
        return model_in, target

    def update_from_bootstrap_batch(
        self,
        bootstrap_batch: mbrl.types.RLEnsembleBatch,
        optimizers: Sequence[torch.optim.Optimizer],
    ):
        if not hasattr(self.model, "members"):
            raise RuntimeError(
                "Model must be ensemble to use `loss_from_bootstrap_batch`."
            )

        model_ins = []
        targets = []
        for i, batch in enumerate(bootstrap_batch):
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            model_ins.append(model_in)
            targets.append(target)
        return self.model.update(model_ins, targets, optimizers)

    def update_from_simple_batch(
        self, batch: mbrl.types.RLBatch, optimizer: torch.optim.Optimizer
    ):
        if hasattr(self.model, "members"):
            raise RuntimeError(
                "Model must not be ensemble to use `loss_from_simple_batch`."
            )

        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.update(model_in, target, optimizer)

    def eval_score_from_simple_batch(self, batch: mbrl.types.RLBatch) -> torch.Tensor:
        model_in, target = self._get_model_input_and_target_from_batch(batch)
        return self.model.eval_score(model_in, target)

    def get_output_and_targets_from_simple_batch(
        self, batch: mbrl.types.RLBatch
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        with torch.no_grad():
            model_in, target = self._get_model_input_and_target_from_batch(batch)
            output = self.model.forward(model_in)
        return output, target

    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        sample=True,
        propagation_method="expectation",
        propagation_indices=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_in = self._get_model_input_from_tensors(obs, actions)
        means, logvars = self.model(
            model_in,
            propagation=propagation_method,
            propagation_indices=propagation_indices,
        )

        if sample:
            assert logvars is not None
            variances = logvars.exp()
            stds = torch.sqrt(variances)
            predictions = torch.normal(means, stds)
        else:
            predictions = means

        next_observs = predictions[:, :-1] if self.learned_rewards else predictions
        if self.target_is_delta:
            tmp_ = next_observs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_observs[:, dim]
            next_observs = tmp_
        rewards = predictions[:, -1:] if self.learned_rewards else None
        return next_observs, rewards

    def save(self, save_dir: Union[str, pathlib.Path]):
        save_dir = pathlib.Path(save_dir)
        self.model.save(str(save_dir / self._MODEL_FNAME))
        if self.normalizer:
            self.normalizer.save(save_dir)

    def load(self, load_dir: Union[str, pathlib.Path]):
        load_dir = pathlib.Path(load_dir)
        self.model.load(str(load_dir / self._MODEL_FNAME))
        if self.normalizer:
            self.normalizer.load(load_dir)


# ------------------------------------------------------------------------ #
# Model trainer
# ------------------------------------------------------------------------ #
class DynamicsModelTrainer:
    def __init__(
        self,
        dynamics_model: DynamicsModelWrapper,
        dataset_train: replay_buffer.IterableReplayBuffer,
        dataset_val: Optional[replay_buffer.IterableReplayBuffer] = None,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        logger: Optional[pytorch_sac.Logger] = None,
        log_frequency: int = 1,
    ):
        self.dynamics_model = dynamics_model
        self.logger = logger
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.log_frequency = log_frequency

        self.optimizers = None
        if self.dynamics_model.model.is_ensemble:
            ensemble = cast(Ensemble, self.dynamics_model.model)
            self.optimizers = []
            for i, model in enumerate(ensemble):
                self.optimizers.append(
                    optim.Adam(
                        model.parameters(), lr=optim_lr, weight_decay=weight_decay
                    )
                )
        else:
            self.optimizers = optim.Adam(
                self.dynamics_model.model.parameters(),
                lr=optim_lr,
                weight_decay=weight_decay,
            )

    # If num_epochs is passed, the function runs for num_epochs. Otherwise trains until
    # `patience` epochs lapse w/o improvement.
    def train(
        self,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = 50,
    ) -> Tuple[List[float], List[float]]:
        update_from_batch_fn = self.dynamics_model.update_from_simple_batch
        if isinstance(self.dynamics_model.model, Ensemble):
            update_from_batch_fn = self.dynamics_model.update_from_bootstrap_batch  # type: ignore
            if not self.dataset_train.is_train_compatible_with_ensemble(
                len(self.dynamics_model.model)
            ):
                raise RuntimeError(
                    "Train dataset is not compatible with ensemble. "
                    "Please use `BootstrapReplayBuffer` class to train ensemble model "
                    "and make sure `buffer.num_members == model.num_members`."
                )

        training_losses, train_eval_scores, val_losses = [], [], []
        best_weights = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        has_val_dataset = (
            self.dataset_val is not None and self.dataset_val.num_stored > 0
        )
        best_val_score = self.evaluate(use_train_set=not has_val_dataset)
        for epoch in epoch_iter:
            avg_losses = []
            for i, bootstrap_batch in enumerate(self.dataset_train):
                batch_size = bootstrap_batch[0][0].shape[0]
                avg_ensemble_loss = update_from_batch_fn(
                    bootstrap_batch, self.optimizers
                )
                # each batch has different size, append avg_ensemble_loss * batch_size
                avg_losses.append(avg_ensemble_loss * batch_size)

            # take the mean of loss over the whole dataset_train
            total_avg_loss = sum(avg_losses) / self.dataset_train.num_stored
            training_losses.append(total_avg_loss)

            train_score = self.evaluate(use_train_set=True)
            train_eval_scores.append(train_score)
            eval_score = train_score
            if has_val_dataset:
                eval_score = self.evaluate()
                val_losses.append(eval_score)

            maybe_best_weights = self.maybe_save_best_weights(
                best_val_score, eval_score
            )
            if maybe_best_weights:
                best_val_score = eval_score
                best_weights = maybe_best_weights
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if self.logger and epoch % self.log_frequency == 0:
                self.logger.log("train/epoch", epoch, epoch)
                self.logger.log("train/model_loss", total_avg_loss, epoch)
                self.logger.log("train/model_score", train_score, epoch)
                self.logger.log("train/model_val_score", eval_score, epoch)
                self.logger.log("train/model_best_val_score", best_val_score, epoch)
                self.logger.dump(epoch, save=True)

            if epochs_since_update >= patience:
                break

        if best_weights:
            self.dynamics_model.model.load_state_dict(best_weights)
        return training_losses, val_losses

    def evaluate(self, use_train_set: bool = False) -> float:
        dataset = self.dataset_val
        if use_train_set:
            if isinstance(self.dataset_train, replay_buffer.BootstrapReplayBuffer):
                self.dataset_train.toggle_bootstrap()
            dataset = self.dataset_train

        total_avg_loss = torch.tensor(0.0)
        for batch in dataset:
            avg_ensemble_loss = self.dynamics_model.eval_score_from_simple_batch(batch)
            total_avg_loss = (
                avg_ensemble_loss.sum() / dataset.num_stored
            ) + total_avg_loss

        if use_train_set and isinstance(
            self.dataset_train, replay_buffer.BootstrapReplayBuffer
        ):
            self.dataset_train.toggle_bootstrap()
        return total_avg_loss.item()

    def maybe_save_best_weights(
        self, best_val_loss: float, val_loss: float
    ) -> Optional[Dict]:
        best_weights = None
        improvement = (
            1 if np.isinf(best_val_loss) else (best_val_loss - val_loss) / best_val_loss
        )
        if improvement > 0.001:
            best_weights = self.dynamics_model.model.state_dict()
        return best_weights


# ------------------------------------------------------------------------ #
# Model environment
# ------------------------------------------------------------------------ #
class ModelEnv:
    def __init__(
        self,
        env: gym.Env,
        model: DynamicsModelWrapper,
        termination_fn,
        reward_fn,
        seed=None,
    ):
        if env.spec and env.spec.id == "kuka-allegro-v0":
            obs_space = env.observation_space
            unwrapped = env.unwrapped

            pos_id = env.state_indices["obj_pose"]["position"].astype(int)
            ori_id = env.state_indices["obj_pose"]["orientation"].astype(int)
            # t_ori_id = env.state_indices["target_orientation"].astype(int)
            EPS = 1e-7

            obj_init_position = torch.Tensor(unwrapped.obj.init_base_position).cuda()
            target_orientation = torch.Tensor([0., 0., 1., 0.]).cuda()

            self.cache = {}

            def normalize(x):
                norm = (x ** 2).sum(-1, keepdim=True) ** 0.5 + 1e-12
                return x / norm

            def compute_theta(obs):
                orientation = obs[..., ori_id]
                # FIXME(poweic): Got norm(target_orientation) > 1 !!!!!
                # target_orientation is part of the states so that a policy can act based on its goal.
                # But it should NOT be part of the state prediction!!! It should be fixed!!
                # target_orientation = obs[..., t_ori_id]

                orientation = normalize(orientation)
                inner_prod = (orientation * target_orientation).sum(dim=-1)
                # assert inner_prod.abs().max() < 1.01
                theta = 2 * torch.acos(inner_prod.abs().clamp(min=0.0, max=1.0 - EPS))
                return theta

            def is_dropped(obs):
                position = obs[..., pos_id]
                diff = position - obj_init_position
                dropped = (diff.abs() > unwrapped.max_dist_consider_dropped).any(dim=-1)
                return dropped

            def _termination_fn(act, next_obs):
                theta = self.cache['theta']
                dropped = self.cache['dropped']
                goal_reached = self.cache['goal_reached']

                done = dropped | goal_reached
                done = done[:, None]

                assert done.ndim == 2 and done.shape[0] == act.shape[0]
                return done

            def _reward_fn(act, next_obs, obs):
                theta = compute_theta(next_obs)
                prev_theta = compute_theta(obs)
                reward = prev_theta - theta

                dropped = is_dropped(next_obs)
                reward -= dropped * unwrapped.dropped_penalty
                reward += (~dropped) * unwrapped.alive_bonus

                goal_reached = theta < unwrapped.proximity_threshold
                reward += goal_reached * unwrapped.success_bonus
                # print (f"goal_reached: {goal_reached.float().mean() * 100:.2f} %")
                # print (f"reward: max = {reward.max()}, min = {reward.min()}, mean = {reward.mean()}")

                self.cache['theta'] = theta
                self.cache['dropped'] = dropped
                self.cache['goal_reached'] = goal_reached

                assert reward.ndim == 1 and reward.shape[0] == act.shape[0]
                return reward

            termination_fn = _termination_fn
            reward_fn = _reward_fn
            print (f"termination_fn: {termination_fn}")
            print (f"reward_fn: {reward_fn}")

        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(seed)
        self._return_as_np = True

    def reset(
        self,
        initial_obs_batch: np.ndarray,
        propagation_method: str = "expectation",
        return_as_np: bool = True,
    ) -> mbrl.types.TensorType:
        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_obs = torch.from_numpy(
            np.copy(initial_obs_batch.astype(np.float32))
        ).to(self.device)

        self._propagation_method = propagation_method
        if propagation_method == "fixed_model":
            assert self.dynamics_model.model.is_ensemble
            self._model_indices = torch.randint(
                len(cast(Ensemble, self.dynamics_model.model)),
                (len(initial_obs_batch),),
                generator=self._rng,
                device=self.device,
            )

        self._return_as_np = return_as_np
        if self._return_as_np:
            return self._current_obs.cpu().numpy()
        return self._current_obs

    def step(self, actions: mbrl.types.TensorType, sample: bool = False):
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            next_observs, pred_rewards = self.dynamics_model.predict(
                self._current_obs,
                actions,
                sample=sample,
                propagation_method=self._propagation_method,
                propagation_indices=self._model_indices,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs, self._current_obs)
            )
            dones = self.termination_fn(actions, next_observs)
            self._current_obs = next_observs
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, {}

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
        propagation_method: str,
    ) -> torch.Tensor:
        assert (
            len(action_sequences.shape) == 3
        )  # population_size, horizon, action_shape
        population_size, horizon, action_dim = action_sequences.shape
        initial_obs_batch = np.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(np.float32)
        self.reset(
            initial_obs_batch, propagation_method=propagation_method, return_as_np=False
        )

        # torch.expand does not allocate new memory and is about 3-4% faster
        # than torch.repeat_interleave(actions_for_step, num_particles, dim=0)
        def repeat_interleave(x, n):
            return x[:, None, :].expand(-1, n, -1).reshape(-1, x.shape[-1])

        total_rewards: torch.Tensor = 0
        for time_step in range(horizon):
            actions_for_step = action_sequences[:, time_step, :]
            action_batch = repeat_interleave(actions_for_step, num_particles)
            _, rewards, _, _ = self.step(action_batch, sample=True)
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return total_rewards.mean(axis=1)
