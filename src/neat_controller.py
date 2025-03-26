import os
import numpy as np
import matplotlib.pyplot as plt
from environment import initialize_simulation
import render
import jax
import jax.numpy as jnp
from tensorneat.problem import BaseProblem
from tensorneat.problem.rl.rl_jit import RLEnv
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG
import render
import time
import warnings
from observations import (
    get_distance_to_target,
    get_joint_positions,
    get_direction_to_target,
)
import logging

#  Parameters
NUM_GENERATIONS = 1
NUM_ARMS = 5
# Update the segments per arm to match NUM_ARMS = 2
NUM_SEGMENTS_PER_ARM = [2, 0, 0, 2, 0]  # Now has 2 elements for 2 arms
SIMULATION_DURATION = 10  # seconds
VISUALIZE_TRAINING = True
SAVE_BEST_EVERY_GEN = 5
TIME_SCALE = 1
FRAME_SKIP = 2
TARGET_DISTANCE = 3.0
SUCCESS_THRESHOLD = 0.75
POPULATION_SIZE = 100
SPECIES_SIZE = 5


best_fitness_history = []
avg_fitness_history = []
generation_count = 0
best_distance_improvement = -float("inf")
best_genome_per_generation = {}


class BrittleStarEnv(RLEnv):
    jitable = True

    def __init__(
        self,
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        target_distance=TARGET_DISTANCE,
        simulation_time=SIMULATION_DURATION,
        time_scale=TIME_SCALE,
        *args,
        **kwargs,
    ):
        super().__init__(max_step=200, *args, **kwargs)

        env, env_state, environment_configuration = initialize_simulation(
            env_type="directed_locomotion",
            num_arms=num_arms,
            num_segments_per_arm=num_segments_per_arm,
            backend="MJX",
            simulation_time=simulation_time,
            time_scale=time_scale,
            target_distance=target_distance,
        )

        self.env = env
        self.initial_env_state = env_state
        self.environment_configuration = environment_configuration
        self.render_fn = jax.jit(env.render)

        self._input_dims, self._output_dims = get_environment_dims(env, env_state)

    def env_step(self, randkey, env_state, action):
        """Step the environment with the given action"""
        #! Scale action values between joint limits
        # Define joint limits
        lower_bounds = jnp.array([-1.047, -0.785] * sum(NUM_SEGMENTS_PER_ARM))
        upper_bounds = jnp.array([1.047, 0.785] * sum(NUM_SEGMENTS_PER_ARM))

        # Normalize between -1 and 1
        normalized_action = jnp.tanh(action)

        action_ranges = (upper_bounds - lower_bounds) / 2
        action_midpoints = (upper_bounds + lower_bounds) / 2
        action = normalized_action * action_ranges + action_midpoints

        next_env_state = self.env.step(state=env_state, action=action)
        joint_positions = []
        for arm in range(NUM_ARMS):
            joint_positions.append(get_joint_positions(next_env_state, arm))

        direction_to_target = get_direction_to_target(next_env_state)
        joint_positions_combined = jnp.concatenate(joint_positions)
        obs = jnp.concatenate([joint_positions_combined, direction_to_target])

        distance = next_env_state.observations["xy_distance_to_target"][0]
        reward = -distance
        done = jnp.array(False)

        info = {}

        return obs, next_env_state, reward, done, info

    def env_reset(self, randkey):
        """Reset the environment"""
        env_state = self.initial_env_state

        joint_positions = []
        for arm in range(NUM_ARMS):
            joint_positions.append(get_joint_positions(env_state, arm))

        direction_to_target = get_direction_to_target(env_state)
        joint_positions_combined = jnp.concatenate(joint_positions)
        obs = jnp.concatenate([joint_positions_combined, direction_to_target])

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




def get_environment_dims(env, state):
    """Get the input and output dimensions from the environment"""
    # Calculate number of inputs and outputs
    num_inputs = sum(
        [len(get_joint_positions(state, arm)) for arm in range(NUM_ARMS)]
    ) + len(get_direction_to_target(state))
    num_outputs = len(env.action_space.sample(rng=jax.random.PRNGKey(seed=0)))
    print(num_inputs, num_outputs)
    return num_inputs, num_outputs


def create_animation(frames, output_path):
    """Create and save an animation from frames"""
    render.create_animation(frames, output_path=output_path)
    print(f"Saved animation to {output_path}")
    # Try to play the video
    render.play_video(output_path)


def train_neat_controller():
    """Train a TensorNEAT controller for the brittle star"""
    print("Starting NEAT training for brittle star locomotion...")

    problem = BrittleStarEnv()  # Use the new RLEnv-based class
    # Get input and output dimensions
    num_inputs, num_outputs = problem._input_dims, problem._output_dims
    print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=POPULATION_SIZE,
            species_size=SPECIES_SIZE,
            survival_threshold=0.2,
            genome=DefaultGenome(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=[ACT.tanh, ACT.relu, ACT.sigmoid],
                    aggregation_options=[AGG.sum],
                ),
                output_transform=ACT.identity,
            ),
        ),
        problem=problem,
        generation_limit=NUM_GENERATIONS,
        fitness_target=10000,
        seed=42,
    )

    print("Initializing TensorNEAT state...")
    state = pipeline.setup()

    state, best_genome = pipeline.auto_run(state)



def example_usage():
    """Example of how to use the framework"""
    print("TensorNEAT Brittle Star Controller")
    print("----------------------------------")

    train_neat_controller()


if __name__ == "__main__":
    example_usage()
