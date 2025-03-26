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
        env, env_state, environment_configuration = initialize_simulation(
            env_type="directed_locomotion",
            num_arms=NUM_ARMS,
            num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
            backend="MJX",
            simulation_time=SIMULATION_DURATION,
            time_scale=TIME_SCALE,
            target_distance=TARGET_DISTANCE,
        )
        self._input_dims, self._output_dims = get_environment_dims(env, env_state)
        self.env = env
        self.env_state = env_state
        self.environment_configuration = environment_configuration
        
    def evaluate(self, state, randkey, act_func, params):
        result,env_state = eval_individual(act_func, state, params, self.env, self.env_state, 
            self.env.step)
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
    jax_obs = jnp.array(obs)
    
    action = act_func(state, params, jax_obs)

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


def get_environment_dims(env, state):
    """Get the input and output dimensions from the environment"""
    # Calculate number of inputs and outputs
    num_inputs = (
        len(state.observations["joint_position"])
        + len(state.observations["joint_velocity"])
        + len(state.observations["disk_position"])
        + len(state.observations["disk_linear_velocity"])
        + len(state.observations["unit_xy_direction_to_target"])
        + len(state.observations["xy_distance_to_target"])
    )
    num_outputs = len(env.action_space.sample(rng=jax.random.PRNGKey(seed=0)))

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
    

    problem = BrittleStarProblem()
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
        fitness_target=10000, # Way to high fitness, I just want to see if it works
        seed=42,
    )
    
    print("Initializing TensorNEAT state...")
    state = pipeline.setup()
    



    pipeline.auto_run(state)
   

def example_usage():
    """Example of how to use the framework"""
    print("TensorNEAT Brittle Star Controller")
    print("----------------------------------")
    
    train_neat_controller()

if __name__ == "__main__":
    example_usage()
