import os
import numpy as np
import matplotlib.pyplot as plt
from environment import initialize_simulation
import render
import jax
import jax.numpy as jnp
from tensorneat.problem import BaseProblem
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG
import time
import warnings

#  Parameters
NUM_GENERATIONS = 50
NUM_ARMS = 2
# Update the segments per arm to match NUM_ARMS = 2
NUM_SEGMENTS_PER_ARM = [1, 1]  # Now has 2 elements for 2 arms
SIMULATION_DURATION = 10  # seconds
VISUALIZE_TRAINING = True
SAVE_BEST_EVERY_GEN = 5
TIME_SCALE = 1
FRAME_SKIP = 2
TARGET_DISTANCE = 3.0
SUCCESS_THRESHOLD = 0.75
POPULATION_SIZE = 100
SPECIES_SIZE = 5

# Create directories for output
os.makedirs("training_progress", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

best_fitness_history = []
avg_fitness_history = []
generation_count = 0
best_distance_improvement = -float("inf")
best_genome_per_generation = {}




class BrittleStarProblem(BaseProblem):
    jitable = True  # necessary for tensorneat

    def __init__(self):
        # Get input and output dimensions from environment
        self._input_dims, self._output_dims = get_environment_dims()

        env, env_state, environment_configuration = initialize_simulation(
            env_type="directed_locomotion",
            num_arms=NUM_ARMS,
            num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
            backend="MJC",
            simulation_time=SIMULATION_DURATION,
            time_scale=TIME_SCALE,
            target_distance=TARGET_DISTANCE,
        )

        self.env = env
        self.env_state = env_state
        self.environment_configuration = environment_configuration
        self.vectorized_step = jax.jit(jax.vmap(env.step))
        
    def evaluate(self, state, randkey, act_func, params):
        result,env_state = eval_individual(act_func, state, params, self.env, self.env_state, 
            self.vectorized_step)
        self.env_state = env_state
        return result 
    
    @property
    def input_shape(self):
        # Return the shape of the observation vector
        return (self._input_dims,)
    
    @property
    def output_shape(self):
        # Return the shape of the action vector
        return (self._output_dims,)
    
    def show(self, state, randkey, act_func, params, *args, **kwargs):
        # Showcase the performance of one individual
        result = eval_individual(act_func, state, params)
        print(f"Fitness: {result['fitness']}")
        print(f"Initial distance: {result['initial_distance']}")
        print(f"Minimum distance: {result['min_distance']}")
        print(f"Distance improvement: {result['distance_improvement']}")
        
        # Create and save animation if needed
        if VISUALIZE_TRAINING and "frames" in result:
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            create_animation(result["frames"], f"{output_dir}/best_individual.mp4")


def eval_individual(act_func, state, params, env, env_state, step):
    """Evaluates a TensorNEAT individual in the brittle star environment"""
    obs = np.concatenate(
            [
                env_state.observations["joint_position"],
                env_state.observations["joint_velocity"],
                env_state.observations["disk_position"],
                env_state.observations["disk_linear_velocity"],
                env_state.observations["unit_xy_direction_to_target"],
                env_state.observations["xy_distance_to_target"],
            ]
        )



    initial_distance_to_target = env_state.observations["xy_distance_to_target"][0]
     # Convert numpy array to jax array for TensorNEAT
    jax_obs = jnp.array(obs)
    
    action = act_func(state, params, jax_obs)
    # print(action)
    #     Traced<ShapedArray(float32[4], weak_type=True)>with<BatchTrace> with
    #   val = Traced<ShapedArray(float32[100,4], weak_type=True)>with<DynamicJaxprTrace>

    env_state = step(state=env_state, action=action)

    curr_distance = env_state.observations["xy_distance_to_target"][0]


    return initial_distance_to_target - curr_distance, env_state



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


def train_neat_controller():
    """Train a TensorNEAT controller for the brittle star"""
    print("Starting NEAT training for brittle star locomotion...")
    
    # Get input and output dimensions
    num_inputs, num_outputs = get_environment_dims()
    print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")
    
    # Create the TensorNEAT pipeline
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
        problem=BrittleStarProblem(),
        generation_limit=NUM_GENERATIONS,
        fitness_target=-1e-4,
        seed=42,
    )
    
    print("Initializing TensorNEAT state...")
    state = pipeline.setup()
    



    pipeline.auto_run(state)
    # ------ The following is lifted from pipeline.auto_run(state) ------ #
    # print("start compile")
    # tic = time.time()
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore",
    #         message=r"The jitted function .* includes a pmap. Using jit-of-pmap can lead to inefficient data movement"
    #     )
    #     compiled_step = jax.jit(pipeline.step).lower(state).compile()

    # if pipeline.show_problem_details:
    #     pipeline.compiled_pop_transform_func = (
    #         jax.jit(jax.vmap(pipeline.algorithm.transform, in_axes=(None, 0)))
    #         .lower(state, pipeline.algorithm.ask(state))
    #         .compile()
    #     )

    # print(
    #     f"compile finished, cost time: {time.time() - tic:.6f}s",
    # )

    # for _ in range(pipeline.generation_limit):

    #     pipeline.generation_timestamp = time.time()

    #     state, previous_pop, fitnesses = compiled_step(state)

    #     fitnesses = jax.device_get(fitnesses)

    #     pipeline.analysis(state, previous_pop, fitnesses)

    #     if max(fitnesses) >= pipeline.fitness_target:
    #         print("Fitness limit reached!")
    #         break

    # if int(state.generation) >= pipeline.generation_limit:
    #     print("Generation limit reached!")

    # if pipeline.is_save:
    #     best_genome = jax.device_get(pipeline.best_genome)
    #     with open(os.path.join(pipeline.genome_dir, f"best_genome.npz"), "wb") as f:
    #         np.savez(
    #             f,
    #             nodes=best_genome[0],
    #             conns=best_genome[1],
    #             fitness=pipeline.best_fitness,
    #         )
    


# def evaluate_saved_genome(checkpoint_path):
#     """Evaluate a saved genome"""
#     print(f"Evaluating genome from {checkpoint_path}...")
    
#     # Get input and output dimensions
#     num_inputs, num_outputs = get_environment_dims()
    
#     # Create the TensorNEAT pipeline with the same configuration as training
#     pipeline = Pipeline(
#         algorithm=NEAT(
#             pop_size=POPULATION_SIZE,
#             species_size=SPECIES_SIZE,
#             survival_threshold=0.2,
#             genome=DefaultGenome(
#                 num_inputs=num_inputs,
#                 num_outputs=num_outputs,
#                 init_hidden_layers=(),
#                 node_gene=BiasNode(
#                     activation_options=[ACT.tanh, ACT.relu, ACT.sigmoid],
#                     aggregation_options=[AGG.sum],
#                 ),
#                 output_transform=ACT.tanh,
#             ),
#         ),
#         problem=BrittleStarProblem(),
#         generation_limit=1,
#         fitness_target=float('inf'),
#         seed=42,
#     )
    
#     # Initialize state
#     state = pipeline.setup()
    
#     # Load the genome
#     genome = pipeline.load_genome(checkpoint_path)
    
#     pipeline.show(state, genome)
    
#     return genome


def example_usage():
    """Example of how to use the framework"""
    print("TensorNEAT Brittle Star Controller")
    print("----------------------------------")
    
    train_neat_controller()


    # # Choose whether to train a new controller or evaluate an existing one
    # choice = input("Enter 't' to train a new controller or 'e' to evaluate a saved one: ")
    
    # if choice.lower() == 't':
    #     train_neat_controller()
    # elif choice.lower() == 'e':
    #     checkpoint_path = input("Enter the path to the checkpoint file: ")
    #     evaluate_saved_genome(checkpoint_path)
    # else:
    #     print("Invalid choice. Exiting.")


if __name__ == "__main__":
    example_usage()
