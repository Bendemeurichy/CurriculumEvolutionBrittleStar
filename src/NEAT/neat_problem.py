from environment import initialize_simulation
import jax
import jax.numpy as jnp

from NEAT.neat_controller import scale_actions_to_joint_limits, extract_observation, calculate_environment_dimensions
import NEAT.config as config
from NEAT.RLEnv import RLEnv

counter = 0


class BrittleStarEnv(RLEnv):
    jitable = True

    def __init__(
        self,
        num_arms=config.NUM_ARMS,
        num_segments_per_arm=config.NUM_SEGMENTS_PER_ARM,
        target_distance=config.TARGET_DISTANCE,
        simulation_time=config.SIMULATION_DURATION,
        time_scale=config.TIME_SCALE,
        *args,
        **kwargs,
    ):
        super().__init__(max_step=config.MAX_STEPS_TRAINING, *args, **kwargs)

        env, env_state, environment_configuration = initialize_simulation(
            env_type="directed_locomotion",
            num_arms=num_arms,
            num_segments_per_arm=num_segments_per_arm,
            backend="MJX",
            simulation_time=simulation_time,
            time_scale=time_scale,
            target_distance=target_distance,
            num_physics_steps_per_control_step=config.NUM_PHYSICS_STEPS_PER_CONTROL_STEP,
            seed=config.SEED,
        )

        self.env = env
        self.initial_env_state = env_state
        self.environment_configuration = environment_configuration
        self.num_segments_per_arm = num_segments_per_arm

        self._input_dims, self._output_dims = calculate_environment_dimensions(env, env_state)

    def env_step(self, _, env_state, action, target, info):
        """Step the environment with the given action"""

        scaled_action = scale_actions_to_joint_limits(
            action, num_segments_per_arm=self.num_segments_per_arm
        )

        next_env_state = self.env.step(state=env_state, action=scaled_action)

        obs = extract_observation(next_env_state, target)

        disk_position = next_env_state.observations["disk_position"][:2]
        distance = jnp.linalg.norm(disk_position - target[:2])
        
        current_velocity = jnp.linalg.norm(
            next_env_state.observations["disk_linear_velocity"][:2]
        )

        # current_velocity = jnp.where(distance < 0.5, current_velocity, 5 * current_velocity)
        
        
        reward = -distance + current_velocity 

        done = jnp.array(distance < config.TARGET_REACHED_THRESHOLD)

        reward = jnp.where(distance < config.TARGET_REACHED_THRESHOLD, reward + 1000.0, reward)
        
        info = {
            "distance": distance,
        }
        
        return obs, next_env_state, reward, done, info

    def env_reset(self, randkey):
        """Reset the environment"""
        # Generate new randkey for target placement
        target_key = jax.random.fold_in(randkey, 0)

        if config.TARGET_POSITION is not None:
            env_state = self.env.reset(
                rng=randkey, target_position=config.TARGET_POSITION
            )
        else:
            # Reset with random target position
            env_state = self.env.reset(rng=target_key)

        # Create separate random keys for each target
        target_key1, target_key2 = jax.random.split(randkey)
        targets = jnp.array(
            [
                self.generate_target_position(target_key1),
                self.generate_target_position(target_key2),
            ]
        )

        obs1 = extract_observation(env_state, targets[0])
        obs2 = extract_observation(env_state, targets[1])
        return (obs1, obs2), env_state, targets

    def generate_target_position(self, rng) -> jnp.ndarray:
        angle = jax.random.uniform(key=rng, shape=(), minval=0, maxval=jnp.pi * 2)
        radius = self.env.environment_configuration.target_distance
        random_position = jnp.array(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), 0.05]
        )
        return random_position

    @property
    def input_shape(self):
        return (self._input_dims,)

    @property
    def output_shape(self):
        return (self._output_dims,)

    def show(
        self,
        state,
        randkey,
        act_func,
        params,
        save_path=None,
        output_type="mp4",
        *args,
        **kwargs,
    ):
        pass
