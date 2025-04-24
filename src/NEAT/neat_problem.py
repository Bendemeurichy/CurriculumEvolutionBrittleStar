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

    def env_step(self, randkey, env_state, action, targets):
        """Step the environment with the given action"""
        scaled_action = scale_actions(action)

        next_env_state = self.env.step(state=env_state, action=scaled_action)

        obs = get_observation(next_env_state)
        
        # Calculate distances to both targets
        disk_position = next_env_state.observations["disk_position"][:2]
        distances = jnp.array([
            jnp.linalg.norm(disk_position - targets[0][:2]),
            jnp.linalg.norm(disk_position - targets[1][:2])
        ])
        
        # Find the index of the closest target
        closest_target_idx = jnp.argmin(distances)
        
        # Get distance to the closest target
        distance = distances[closest_target_idx]
        
        # Calculate previous distances for progress calculation
        prev_disk_position = env_state.observations["disk_position"][:2]
        prev_distances = jnp.array([
            jnp.linalg.norm(prev_disk_position - targets[0][:2]),
            jnp.linalg.norm(prev_disk_position - targets[1][:2])
        ])
        
        # Get previous distance to the same target (not necessarily the previously closest one)
        prev_distance = prev_distances[closest_target_idx]
        
        # Calculate progress toward target (positive when getting closer)
        progress = prev_distance - distance
        
        # Calculate total movement (to prevent staying still)
        disk_velocity = jnp.linalg.norm(next_env_state.observations["disk_linear_velocity"][:2])
        
        # Calculate energy usage (sum of absolute actuator forces)
        actuator_forces = next_env_state.observations["actuator_force"]
        energy_usage = jnp.sum(jnp.abs(actuator_forces))
        
        # Energy efficiency penalty - scaled to be significant but not overwhelming
        energy_penalty = energy_usage * 0.01
        
        # Combined reward: distance penalty + progress reward + movement incentive - energy penalty
        reward = -distance + progress * 3.0 + jnp.minimum(disk_velocity, 0.5) * 0.2 - energy_penalty
        
        # Bonus reward for getting very close to target
        reward = jnp.where(distance < 0.5, reward + (0.5 - distance) * 5.0, reward)
        
        done = jnp.array(distance < 0.1)

        info = {"closest_target_idx": closest_target_idx}

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
        
        # Create separate random keys for each target
        target_key1, target_key2 = jax.random.split(randkey)
        targets = jnp.array([
            self.generate_target_position(target_key1), 
            self.generate_target_position(target_key2)
        ])
        
        return obs, env_state, targets
    
    def generate_target_position(self,rng) -> jnp.ndarray:
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