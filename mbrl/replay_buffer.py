from typing import List, Sized, Tuple, Union

import numpy as np


class SimpleReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        obs_type=np.float32,
        action_type=np.float32,
    ):
        self.obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.action = np.empty((capacity, *action_shape), dtype=action_type)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.bool)
        self.cur_idx = 0
        self.capacity = capacity
        self.num_stored = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ):
        self.obs[self.cur_idx] = obs
        self.next_obs[self.cur_idx] = next_obs
        self.action[self.cur_idx] = action
        self.reward[self.cur_idx] = reward
        self.done[self.cur_idx] = done

        self.cur_idx = (self.cur_idx + 1) % self.capacity
        self.num_stored = min(self.num_stored + 1, self.capacity)

    def sample(self, batch_size: int) -> Sized:
        indices = np.random.choice(self.num_stored, size=batch_size)
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: Sized) -> Tuple:
        obs = self.obs[indices]
        next_obs = self.next_obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        done = self.done[indices]

        return obs, action, next_obs, reward, done

    def __len__(self):
        return self.num_stored

    def save(self, path: str):
        np.savez(
            path,
            obs=self.obs[: self.num_stored],
            next_obs=self.next_obs[: self.num_stored],
            action=self.action[: self.num_stored],
            reward=self.reward[: self.num_stored],
            done=self.done[: self.num_stored],
        )

    def load(self, path: str):
        data = np.load(path)
        num_stored = len(data["obs"])
        self.obs[:num_stored] = data["obs"]
        self.next_obs[:num_stored] = data["next_obs"]
        self.action[:num_stored] = data["action"]
        self.reward[:num_stored] = data["reward"]
        self.done[:num_stored] = data["done"]
        self.num_stored = num_stored
        self.cur_idx = self.num_stored % self.capacity


class IterableReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        obs_type=np.float32,
        action_type=np.float32,
        shuffle_each_epoch: bool = False,
    ):
        super(IterableReplayBuffer, self).__init__(
            capacity,
            obs_shape,
            action_shape,
            obs_type=obs_type,
            action_type=action_type,
        )
        self.batch_size = batch_size
        self._current_batch = 0
        self._order: np.ndarray = np.arange(self.capacity)
        self._shuffle_each_epoch = shuffle_each_epoch

    def _get_indices_next_batch(self) -> Sized:
        start_idx = self._current_batch * self.batch_size
        if start_idx >= self.num_stored:
            raise StopIteration
        end_idx = min((self._current_batch + 1) * self.batch_size, self.num_stored)
        order_indices = range(start_idx, end_idx)
        indices = self._order[order_indices]
        self._current_batch += 1
        return indices

    def __iter__(self):
        self._current_batch = 0
        if self._shuffle_each_epoch:
            self._order = np.random.permutation(self.num_stored)
        return self

    def __next__(self):
        return self._batch_from_indices(self._get_indices_next_batch())

    def __len__(self):
        return (self.num_stored - 1) // self.batch_size + 1

    def load(self, path: str):
        super().load(path)
        self._current_batch = 0


# TODO Add a transition type to encapsulate this batch data
class BootstrapReplayBuffer(IterableReplayBuffer):
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        num_members: int,
        obs_shape: Tuple[int],
        action_shape: Tuple[int],
        obs_type=np.float32,
        action_type=np.float32,
        shuffle_each_epoch: bool = False,
    ):
        super(BootstrapReplayBuffer, self).__init__(
            capacity,
            batch_size,
            obs_shape,
            action_shape,
            obs_type=obs_type,
            action_type=action_type,
            shuffle_each_epoch=shuffle_each_epoch,
        )
        self.member_indices: List[List[int]] = [None for _ in range(num_members)]

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ):
        super().add(obs, action, next_obs, reward, done)

    def __iter__(self):
        super().__iter__()
        for i in range(len(self.member_indices)):
            self.member_indices[i] = np.random.choice(
                self.num_stored, size=self.num_stored, replace=True
            )
        return self

    def __next__(self):
        indices = self._get_indices_next_batch()
        batches = []
        for member_idx in self.member_indices:
            content_indices = member_idx[indices]
            batches.append(self._batch_from_indices(content_indices))
        return batches

    def sample(self, batch_size: int, ensemble=True) -> Union[List[Tuple], Tuple]:
        if ensemble:
            batches = []
            for member_idx in self.member_indices:
                indices = np.random.choice(self.num_stored, size=batch_size)
                content_indices = member_idx[indices]
                batches.append(self._batch_from_indices(content_indices))
            return batches
        else:
            indices = np.random.choice(self.num_stored, size=batch_size)
            return self._batch_from_indices(indices)
