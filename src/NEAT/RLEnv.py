from typing import Callable

import jax
from jax import vmap, numpy as jnp
import numpy as np

from tensorneat.problem.base import BaseProblem
from tensorneat.common import State


class RLEnv(BaseProblem):
    jitable = True

    def __init__(
        self,
        max_step=1000,
        repeat_times=1,
        action_policy: Callable = None,
        obs_normalization: bool = False,
        sample_policy: Callable = None,
        sample_episodes: int = 0,
    ):
        """
        action_policy take three args:
            randkey, forward_func, obs
            randkey is a random key for jax.random
            forward_func is a function which receive obs and return action forward_func(obs) - > action
            obs is the observation of the environment

        sample_policy take two args:
            randkey, obs -> action
        """

        super().__init__()
        self.max_step = max_step
        self.repeat_times = repeat_times
        self.action_policy = action_policy

        if obs_normalization:
            assert sample_policy is not None, "sample_policy must be provided"
            assert sample_episodes > 0, "sample_size must be greater than 0"
            self.sample_policy = sample_policy
            self.sample_episodes = sample_episodes
        self.obs_normalization = obs_normalization

    def setup(self, state=State()):
        print("Called setup!")
        if self.obs_normalization:
            print("Sampling episodes for normalization")
            keys = jax.random.split(state.randkey, self.sample_episodes)
            dummy_act_func = (
                lambda s, p, o: o
            )  # receive state, params, obs and return the original obs
            dummy_sample_func = lambda rk, act_func, obs: self.sample_policy(
                rk, obs
            )  # ignore act_func

            def sample(rk):
                return self._evaluate_once(
                    state, rk, dummy_act_func, None, dummy_sample_func, True
                )

            rewards, episodes = jax.jit(vmap(sample))(keys)

            obs = jax.device_get(
                episodes["obs"]
            )  # shape: (sample_episodes, max_step, *input_shape)
            obs = obs.reshape(
                -1, *self.input_shape
            )  # shape: (sample_episodes * max_step, *input_shape)

            obs_axis = tuple(range(obs.ndim))
            valid_data_flag = np.all(~jnp.isnan(obs), axis=obs_axis[1:])
            obs = obs[valid_data_flag]

            obs_mean = np.mean(obs, axis=0)
            obs_std = np.std(obs, axis=0)

            state = state.register(
                problem_obs_mean=obs_mean,
                problem_obs_std=obs_std,
            )

            print("Sampling episodes for normalization finished.")
            print("valid data count: ", obs.shape[0])
            print("obs_mean: ", obs_mean)
            print("obs_std: ", obs_std)
        state = state.register(current_generation=0)
        return state

    def evaluate(self, state: State, randkey, act_func: Callable, params, shared_key):
        keys = jax.random.split(randkey, self.repeat_times)
        jax.debug.print("Evaluating...")
        # increment state.current_generation

        target_key1, target_key2 = jax.random.split(shared_key)
        targets = jnp.array(
            [
                self.generate_target_position(target_key1),
                self.generate_target_position(target_key2),
            ]
        )

        rewards = vmap(
            self._evaluate_once, in_axes=(None, 0, None, None, None, None, None, None)
        )(
            state,
            keys,
            act_func,
            params,
            self.action_policy,
            False,
            targets,
            self.obs_normalization,
        )

        return rewards.mean()

    def _evaluate_once(
        self,
        state,
        randkey,
        act_func,
        params,
        action_policy,
        record_episode,
        targets,
        normalize_obs=False,
    ):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_obs, init_env_state, _ = self.reset(rng_reset)

        if record_episode:
            obs_array = jnp.full((self.max_step, *self.input_shape), jnp.nan)
            action_array = jnp.full((self.max_step, *self.output_shape), jnp.nan)
            reward_array = jnp.full((self.max_step,), jnp.nan)
            episode = {
                "obs": obs_array,
                "action": action_array,
                "reward": reward_array,
            }
        else:
            episode = None

        def cond_func(carry):
            _, _, _, done, _, count, _, rk, _, _ = carry
            return ~done & (count < self.max_step)

        def body_func(carry):
            (obs, env_state, rng, done, tr, count, epis, rk, target, info) = (
                carry  # tr -> total reward; rk -> randkey
            )

            if normalize_obs:
                obs = norm_obs(state, obs)

            if action_policy is not None:
                forward_func = lambda obs: act_func(state, params, obs)
                action = action_policy(rk, forward_func, obs)
            else:
                action = act_func(state, params, obs)
            next_obs, next_env_state, reward, done, info = self.step(
                rng, env_state, action, target, info
            )

            next_rng, _ = jax.random.split(rng)

            if record_episode:
                epis["obs"] = epis["obs"].at[count].set(obs)
                epis["action"] = epis["action"].at[count].set(action)
                epis["reward"] = epis["reward"].at[count].set(reward)
            return (
                next_obs,
                next_env_state,
                next_rng,
                done,
                tr + reward,
                count + 1,
                epis,
                jax.random.split(rk)[0],
                target,
                info,
            )

        # Also update the tuple unpacking to handle all 9 returned elements
        _, _, _, _, total_reward, _, _, _, _, info = jax.lax.while_loop(
            cond_func,
            body_func,
            (init_obs[0], init_env_state, rng_episode, False, 0.0, 0, episode, randkey, targets[0], {
                "no_movement_count": 0,
                "prev_arm_orientations": jnp.zeros(5),  # Initial arm orientations
                "distance": 0.0,
                "progress": 0.0,
                "positioning_activity": 0.0
            }),
        )

        # Update the second while_loop unpacking as well
        _, _, _, _, total_reward2, _, _, _, _, _ = jax.lax.while_loop(
            cond_func,
            body_func,
            (init_obs[1], init_env_state, rng_episode, False, 0.0, 0, episode, randkey, targets[1], {
                "no_movement_count": 0,
                "prev_arm_orientations": jnp.zeros(5),  # Initial arm orientations
                "distance": 0.0,
                "progress": 0.0,
                "positioning_activity": 0.0,
            }),
        )
        total_reward = jnp.minimum(total_reward2, total_reward2)

        if record_episode:
            return total_reward, episode
        else:
            return total_reward

    def step(self, randkey, env_state, action, targets, info):
        return self.env_step(randkey, env_state, action, targets, info)

    def reset(self, randkey):
        return self.env_reset(randkey)

    def env_step(self, randkey, env_state, action):
        raise NotImplementedError

    def env_reset(self, randkey):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        raise NotImplementedError

    def get_observation(self, env_state):
        """Get observation from environment state"""
        # This is a simplified version - adjust according to your actual observation logic
        return env_state.observations

    def compute_distance(self, state, params, act_func):
        randkey = state.randkey
        (obs, _), env_state, targets = self.env_reset(randkey)
        target = targets[0]

        def cond_fn(carry):
            obs, env_state, step_count, done, info = carry
            return (~done) & (step_count < self.max_step)

        def body_fn(carry):
            obs, env_state, step_count, done, info = carry
            action = act_func(state, params, obs)
            obs, env_state, reward, done, info = self.env_step(
                randkey, env_state, action, target, info
            )
            return obs, env_state, step_count + 1, done, info

        # Initial carry
        info = {
            "no_movement_count": jnp.zeros((), dtype=jnp.int32),
            "prev_arm_orientations": jnp.zeros(5),
            "distance": jnp.array(0.0),
            "progress": jnp.array(0.0),
            "positioning_activity": jnp.array(0.0),
        }
        done = jnp.array(False)
        step_count = jnp.array(0)

        obs, env_state, step_count, done, info = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (obs, env_state, step_count, done, info),
        )

        return info["distance"]


def norm_obs(state, obs):
    return (obs - state.problem_obs_mean) / (state.problem_obs_std + 1e-6)
