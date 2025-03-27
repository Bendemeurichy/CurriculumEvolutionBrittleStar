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
import time
import warnings
from observations import (
    get_distance_to_target,
    get_joint_positions,
    get_direction_to_target,
)

#  Parameters
NUM_GENERATIONS = 10
NUM_ARMS = 5
# Update the segments per arm to match NUM_ARMS = 2
NUM_SEGMENTS_PER_ARM = [2, 0, 0, 2, 0]  # Now has 2 elements for 2 arms
SIMULATION_DURATION = 40  # seconds
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


def scale_actions(actions):
    """Scale actions to match joint limits"""
    lower_bounds = jnp.array([-1.047, -0.785] * sum(NUM_SEGMENTS_PER_ARM))
    upper_bounds = jnp.array([1.047, 0.785] * sum(NUM_SEGMENTS_PER_ARM))

    # Normalize between -1 and 1
    normalized_action = jnp.tanh(actions)

    action_ranges = (upper_bounds - lower_bounds) / 2
    action_midpoints = (upper_bounds + lower_bounds) / 2
    scaled_action = normalized_action * action_ranges + action_midpoints

    return scaled_action


def get_observation(env_state):
    """Extract observation from environment state"""
    joint_positions = []
    for arm in range(NUM_ARMS):
        joint_positions.append(get_joint_positions(env_state, arm))

    direction_to_target = get_direction_to_target(env_state)
    joint_positions_combined = jnp.concatenate(joint_positions)
    obs = jnp.concatenate([joint_positions_combined, direction_to_target])

    return obs


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
        # Scale the action
        scaled_action = scale_actions(action)

        # Step the environment
        next_env_state = self.env.step(state=env_state, action=scaled_action)

        # Get observation
        obs = get_observation(next_env_state)

        # Calculate reward
        distance = next_env_state.observations["xy_distance_to_target"][0]
        reward = distance
        done = jnp.array(False)

        info = {}

        return obs, next_env_state, reward, done, info

    def env_reset(self, randkey):
        """Reset the environment"""
        env_state = self.initial_env_state
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
    # Make sure directory exists for output path
    directory = os.path.dirname(output_path)
    if directory:  # Only try to create if there's a directory component
        os.makedirs(directory, exist_ok=True)

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
        fitness_target=1e-2,
        seed=42,
        is_save=True,
        save_dir="output",
    )

    print("Initializing TensorNEAT state...")
    state = pipeline.setup()

    state, best_genome = pipeline.auto_run(state)

    if best_genome is not None:
        print("\nTraining complete! Visualizing best individual...")
        # Make sure to provide a proper path with directory
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "training_progress.mp4")
        visualize_brittlestar(
            state, best_genome, pipeline.algorithm, save_path=save_path
        )

        visualize_brittlestar(
            state, best_genome, pipeline.algorithm, save_path=save_path
        )

    return state, best_genome


def visualize_brittlestar(state, genome, algorithm, save_path=None):
    """Visualize the trained brittlestar model"""
    print("Creating visualization...")

    # Create environment for visualization
    env, env_state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        backend="MJC",
        simulation_time=SIMULATION_DURATION,
        time_scale=TIME_SCALE,
        target_distance=TARGET_DISTANCE,
    )

    # Transform the genome for use with forward function
    transformed_genome = algorithm.transform(state, genome)

    # Set up parameters for visualization
    max_steps = 200
    frames = []

    # Get initial observation
    obs = get_observation(env_state)

    # Store initial distance for reporting
    initial_distance = env_state.observations["xy_distance_to_target"][0]
    min_distance = initial_distance
    total_reward = 0.0

    # Random key for actions
    rng_key = jax.random.PRNGKey(42)

    print("Running simulation...")
    for step in range(max_steps):
        # Render current state
        # Should zoom camera 0
        camera_id = 0
        env._env._mj_model.cam_pos[camera_id] *= 0.5  # Move camera closer
        frame = env.render(env_state)
        processed_frame = render.post_render(
            render_output=frame, environment_configuration=environment_configuration
        )
        frames.append(processed_frame)

        # Get action using the neural network
        rng_key, subkey = jax.random.split(rng_key)
        action = algorithm.forward(state, transformed_genome, obs)

        # Scale and apply the action
        scaled_action = scale_actions(action)

        # Step the environment
        env_state = env.step(state=env_state, action=scaled_action)

        # Update observation
        obs = get_observation(env_state)

        # Get current distance and update minimum
        current_distance = env_state.observations["xy_distance_to_target"][0]
        min_distance = min(float(min_distance), float(current_distance))

        # Update reward
        reward = -current_distance
        total_reward += reward

        # Print progress
        print(
            f"Step {step}: Distance = {current_distance:.4f}, Min Distance = {min_distance:.4f}"
        )

        print(
            f"Step {step}: Distance = {current_distance:.4f}, Min Distance = {min_distance:.4f}"
        )

    # Display results
    print("\n=== Visualization Results ===")
    print(f"Initial distance: {initial_distance:.4f}")
    print(f"Minimum distance: {min_distance:.4f}")
    print(f"Distance improvement: {initial_distance - min_distance:.4f}")
    print(f"Total reward: {total_reward:.4f}")

    # Save video - ensure we have a valid path with directory
    if save_path is None:
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "brittlestar_visualization.mp4")
    else:
        # Make sure directory exists for the provided path
        directory = os.path.dirname(save_path)
        if directory:  # Only try to create if there's a directory component
            os.makedirs(directory, exist_ok=True)

    create_animation(frames, save_path)

    return {
        "frames": frames,
        "fitness": initial_distance - min_distance,
        "initial_distance": initial_distance,
        "min_distance": min_distance,
        "distance_improvement": initial_distance - min_distance,
        "total_reward": total_reward,
        "frames": frames,
        "fitness": initial_distance - min_distance,
        "initial_distance": initial_distance,
        "min_distance": min_distance,
        "distance_improvement": initial_distance - min_distance,
        "total_reward": total_reward,
    }


def example_usage():
    """Example of how to use the framework"""
    print("TensorNEAT Brittle Star Controller")
    print("----------------------------------")

    state, best_genome = train_neat_controller()

    # You can also visualize separately after training
    # visualize_brittlestar(state, best_genome, "custom_visualization.mp4")


if __name__ == "__main__":
    example_usage()
