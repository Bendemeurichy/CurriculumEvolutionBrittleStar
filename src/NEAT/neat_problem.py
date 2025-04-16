from environment import initialize_simulation
import jax
import jax.numpy as jnp
# from tensorneat.problem.rl.rl_jit import RLEnv
from NEAT.neat_controller import scale_actions, get_observation, get_environment_dims

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
            seed=config.SEED
        )

        self.env = env
        self.initial_env_state = env_state
        self.environment_configuration = environment_configuration

        self._input_dims, self._output_dims = get_environment_dims(env, env_state)

    def env_step(self, randkey, env_state, action):
        """Step the environment with the given action"""
        scaled_action = scale_actions(action)

        next_env_state = self.env.step(state=env_state, action=scaled_action)

        obs = get_observation(next_env_state)

        distance = next_env_state.observations["xy_distance_to_target"][0]
        
        # Get the previous distance for progress calculation
        prev_distance = env_state.observations["xy_distance_to_target"][0]
        
        # Calculate progress toward target (positive when getting closer)
        progress = prev_distance - distance
        
        # Calculate total movement (to prevent staying still)
        disk_velocity = jnp.linalg.norm(next_env_state.observations["disk_linear_velocity"][:2])
        
        # Combined reward: distance penalty + progress reward + small movement incentive
        reward = -distance + progress * 3.0 + jnp.minimum(disk_velocity, 0.5) * 0.2
        
        # Bonus reward for getting very close to target
        reward = jnp.where(distance < 0.5, reward + (0.5 - distance) * 5.0, reward)
        
        done = jnp.array(distance < 0.1)

        info = {}

        return obs, next_env_state, reward, done, info

    def env_reset(self, randkey):
        """Reset the environment"""
        # Generate new randkey for target placement
        target_key = jax.random.fold_in(randkey, 0)
        
        if config.TARGET_POSITION is not None:
            env_state = self.env.reset(rng=randkey, target_position=config.TARGET_POSITION)
        else:
            # Reset with random target position
            env_state = self.env.reset(rng=target_key)
            

        obs = get_observation(env_state)
        return obs, env_state

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