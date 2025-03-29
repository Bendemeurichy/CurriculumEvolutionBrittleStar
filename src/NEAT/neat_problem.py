
from environment import initialize_simulation
import jax
import jax.numpy as jnp
from tensorneat.problem.rl.rl_jit import RLEnv
from NEAT.neat_controller import scale_actions, get_observation, get_environment_dims

import NEAT.config as config 


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
        reward = -distance # don't remove the '-' here cuz we want to minimize the distance. 
                           # The reason why it 'worked' without the - is bcs visulization & training uses different targets and it happend to be about the other side of the circle I think
        done = jnp.array(False)

        info = {}

        return obs, next_env_state, reward, done, info

    def env_reset(self, randkey):
        """Reset the environment"""
        #env_state = self.env.reset(rng=randkey)
        env_state = self.initial_env_state
        if config.TARGET_POSITION is not None:
            env_state = self.env.reset(rng=randkey, target_position=config.TARGET_POSITION)

        # target_id = env_state.mj_model.body("target").id
        # target = env_state.mjx_data.xpos[target_id]
        # jax.debug.print("({}, {})", target[0], target[1])


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