from environment import initialize_simulation
import jax
import jax.numpy as jnp

# from tensorneat.problem.rl.rl_jit import RLEnv
from NEAT.neat_controller import scale_actions, get_observation, get_environment_dims
from observations import get_disk_position
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

        self._input_dims, self._output_dims = get_environment_dims(env, env_state)

    def env_step(self, randkey, env_state, action, target, info):
        """Step the environment with the given action"""

        scaled_action = scale_actions(
            action, num_segments_per_arm=self.num_segments_per_arm
        )

        next_env_state = self.env.step(state=env_state, action=scaled_action)

        obs = get_observation(next_env_state, target)

        # Calculate key metrics
        disk_position = next_env_state.observations["disk_position"][:2]
        distance = jnp.linalg.norm(disk_position - target[:2])
        
        #prev_disk_position = env_state.observations["disk_position"][:2]
        #prev_distance = jnp.linalg.norm(prev_disk_position - target[:2])
        
        # Calculate progress toward target
        # progress = prev_distance - distance
        
        # Calculate movement
        current_velocity = jnp.linalg.norm(
            next_env_state.observations["disk_linear_velocity"][:2]
        )
        
        # # Update velocity history
        # velocity_history = info.get("velocity_history", jnp.zeros(30))
        # # Shift history and add current velocity
        # velocity_history = jnp.roll(velocity_history, -1)
        # velocity_history = velocity_history.at[-1].set(current_velocity)
        
        # # Calculate average velocity over history
        # avg_velocity = jnp.mean(velocity_history)
        
        # # Track no movement for stuck detection
        # no_movement_count = info["no_movement_count"]
        # no_movement_count = jax.lax.cond(
        #     current_velocity < 0.05,
        #     lambda _: no_movement_count + 1,
        #     lambda _: no_movement_count - 5,
        #     operand=None,
        # )

        # no_movement_count = jnp.maximum(no_movement_count, 0)
                
        # Use average velocity in reward instead of instantaneous velocity
        reward = -distance + current_velocity 
        
        # Terminal condition
        done = jnp.array(distance < 0.1)
        reward = jnp.where(distance < 0.1, reward + 500.0, reward)
        
        # Update info with metrics including velocity history
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

        obs1 = get_observation(env_state, targets[0])
        obs2 = get_observation(env_state, targets[1])
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
