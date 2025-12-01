"""
Environment Wrappers for FPO Compatibility

The mujoco_playground locomotion environments use dict-type observations
(privileged_state, state) while FPO expects a single flat observation vector.

This module provides wrappers to flatten the observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any


class FlatObsEnvWrapper:
    """Wrapper that flattens dict observations to a single vector.

    Uses the 'state' key by default (non-privileged observations).
    """

    def __init__(self, env, obs_key: str = "state"):
        self._env = env
        self._obs_key = obs_key

        # Calculate flat observation size
        obs_size_dict = env.observation_size
        if isinstance(obs_size_dict, dict):
            self._observation_size = obs_size_dict[obs_key][0]
        else:
            self._observation_size = obs_size_dict

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self._env.action_size

    @property
    def dt(self) -> float:
        return self._env.dt

    @property
    def sys(self):
        return self._env.sys

    def reset(self, rng):
        state = self._env.reset(rng)
        return self._flatten_state(state)

    def step(self, state, action):
        next_state = self._env.step(state, action)
        return self._flatten_state(next_state)

    def _flatten_state(self, state):
        """Flatten the observation in the state."""
        obs = state.obs
        if isinstance(obs, dict):
            flat_obs = obs[self._obs_key]
        else:
            flat_obs = obs

        # Create new state with flat observation
        return state.replace(obs=flat_obs)

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment."""
        return getattr(self._env, name)


def wrap_env_for_fpo(env, obs_key: str = "state"):
    """Wrap an environment to be compatible with FPO.

    Args:
        env: The environment to wrap
        obs_key: Which observation key to use ('state' or 'privileged_state')

    Returns:
        Wrapped environment with flat observation space
    """
    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        return FlatObsEnvWrapper(env, obs_key=obs_key)
    return env
