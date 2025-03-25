import os
import numpy as np
import matplotlib.pyplot as plt
from environment import initialize_simulation
import render


#  Parameters
NUM_GENERATIONS = 50
NUM_ARMS = 4
NUM_SEGMENTS_PER_ARM = [1, 0, 0, 1, 0]
SIMULATION_DURATION = 10  # seconds
VISUALIZE_TRAINING = True
SAVE_BEST_EVERY_GEN = 5
TIME_SCALE = 1
FRAME_SKIP = 2
TARGET_DISTANCE = 3.0
SUCCESS_THRESHOLD = 0.75

# Create directories for output
os.makedirs("training_progress", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

best_fitness_history = []
avg_fitness_history = []
generation_count = 0
best_distance_improvement = -float("inf")
best_genome_per_generation = {}


def eval_individual(individual, create_network_func):
    net = create_network_func(individual)

    env, state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        backend="MJC",
        simulation_time=SIMULATION_DURATION,
        time_scale=TIME_SCALE,
        target_distance=TARGET_DISTANCE,
    )

    initial_distance_to_target = state.observations["xy_distance_to_target"][0]
    frames = []

    distances = [initial_distance_to_target]

    frame = env.render(state=state)
    processed_frame = render.post_render(
        render_output=frame, environment_configuration=environment_configuration
    )
    frames.append(processed_frame)

    steps = 0
    max_steps = int(
        SIMULATION_DURATION
        / (
            environment_configuration.time_scale
            * 0.002
            * environment_configuration.num_physics_steps_per_control_step
        )
    )

    while steps < max_steps:
        # Use neural network to determine action based on observations
        # FIXME: Remove/add some observations, don't forget to update `get_environment_dims` as well!!!
        obs = np.concatenate(
            [
                state.observations["joint_position"],
                state.observations["joint_velocity"],
                state.observations["disk_position"],
                state.observations["disk_linear_velocity"],
                state.observations["unit_xy_direction_to_target"],
                state.observations["xy_distance_to_target"],
            ]
        )

        # TODO: this action is the output of the neat network
        action = net(obs)
        action = np.array(action)

        old_state = state  # Save previous state
        state = env.step(state=state, action=action)
        steps += 1

        if state.terminated and steps < max_steps:
            current_distance = state.observations["xy_distance_to_target"][0]
            print(
                f"WARNING: Environment reports reaching target at distance: {current_distance:.3f}"
            )
            print(
                f"Success threshold would be: {TARGET_DISTANCE * (1 - SUCCESS_THRESHOLD):.3f}"
            )

            distances.append(current_distance)

            state = old_state
            state.terminated = False
            state.truncated = False
        else:
            current_distance = state.observations["xy_distance_to_target"][0]
            distances.append(current_distance)

        if steps % FRAME_SKIP == 0:
            processed_frame = render.post_render(
                env.render(state=state), environment_configuration
            )
            frames.append(processed_frame)

        if steps >= max_steps:
            break

    # ------- Calculate fitness ------- #
    min_distance = min(distances)
    distance_improvement = initial_distance_to_target - min_distance

    avg_final_distance = (
        np.mean(distances[-20:]) if len(distances) >= 20 else np.mean(distances)
    )
    stability_bonus = max(0, initial_distance_to_target - avg_final_distance) * 2

    # Bonus for consistent movement toward target
    movement_consistency = 0
    for i in range(1, len(distances)):
        if distances[i] < distances[i - 1]:  # Moving toward target
            movement_consistency += 0.1

    # Calculate final fitness (prioritize moving toward target)
    fitness = (distance_improvement * 10.0) + stability_bonus + movement_consistency

    # Store additional data
    result = {
        "fitness": fitness,
        "frames": frames,
        "distance_improvement": distance_improvement,
        "min_distance": min_distance,
        "initial_distance": initial_distance_to_target,
    }

    return result


def create_fitness_plot(generation, avg_fitness, best_fitness):
    """Create a plot showing fitness progress over generations"""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(generation + 1), avg_fitness, "b-", label="Average Fitness")
    plt.plot(range(generation + 1), best_fitness, "r-", label="Best Fitness")
    plt.grid(True)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Training Progress - Generation {generation}")
    plt.legend()
    return fig


def get_environment_dims():
    """Get the input and output dimensions from the environment"""
    env, state, _ = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        target_distance=TARGET_DISTANCE,
    )

    # Calculate number of inputs and outputs
    num_inputs = (
        len(state.observations["joint_position"])
        + len(state.observations["joint_velocity"])
        + len(state.observations["disk_position"])
        + len(state.observations["disk_linear_velocity"])
        + len(state.observations["unit_xy_direction_to_target"])
        + len(state.observations["xy_distance_to_target"])
    )

    num_outputs = len(env.action_space.sample())

    return num_inputs, num_outputs


def create_animation(frames, output_path):
    """Create and save an animation from frames"""
    render.create_animation(frames, output_path=output_path)
    print(f"Saved animation to {output_path}")
    # Try to play the video
    render.play_video(output_path)


class DummyNetwork:
    """A placeholder network that returns random actions"""

    def __init__(self):
        pass

    def __call__(self, observations):
        """Return random actions"""

        # FIXME: right now we just select random actions, if we use neat then the observations should be go through the neural network
        return np.random.uniform(-1, 1, 2 * NUM_ARMS * NUM_SEGMENTS_PER_ARM)


def example_usage():
    """Exampl of how to use the framework"""
    print("This is a boilerplate for NEAT-based controllers.")
    print("Implement a concrete NEAT controller class before using.")

    num_inputs, num_outputs = get_environment_dims()
    print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

    # Example evaluation with dummy network, this should become the tensorneat thing
    dummy_network = DummyNetwork()
    dummy_individual = "idk"  # Placeholder for genome, this should become the genome object. 1 genome is 1 neural network I think

    def dummy_network_creator(genome):
        # This function would normally create a network from a genome
        # TODO: create 1 neat network for a given genome
        return dummy_network

    print("Evaluating with random actions as a test:")
    result = eval_individual(dummy_individual, dummy_network_creator)

    print(f"Random actions achieved fitness: {result['fitness']}")
    print(f"Initial distance: {result['initial_distance']}")
    print(f"Minimum distance: {result['min_distance']}")

    # Example of creating and saving animation
    if VISUALIZE_TRAINING:
        output_dir = "example_output"
        os.makedirs(output_dir, exist_ok=True)
        create_animation(result["frames"], f"{output_dir}/random_actions.mp4")


if __name__ == "__main__":
    example_usage()
