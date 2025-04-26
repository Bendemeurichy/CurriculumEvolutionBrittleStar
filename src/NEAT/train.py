import pickle
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# from NEAT.genome_extension import extend_genome
from NEAT.neat_problem import BrittleStarEnv
from NEAT.neat_controller import init_pipeline
from NEAT.visualize import load_model


def save_genome(best, output_dir="./", filename="best_genome.pkl"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, filename)
    with open(model_path, "wb") as f:
        pickle.dump(best, f)
    print(f"Best genome saved to {model_path}")


def train_neat_controller():
    """Train a TensorNEAT controller for the brittle star"""
    print("Starting NEAT training for brittle star locomotion...")

    problem = BrittleStarEnv()  # Use the new RLEnv-based class
    # Get input and output dimensions
    num_inputs, num_outputs = problem._input_dims, problem._output_dims
    print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

    pipeline = init_pipeline(problem)

    print("Initializing TensorNEAT state...")
    state = pipeline.setup()

    # genome = load_model("./models/best_genome.pkl")

    # state = extend_genome(
    #     state,
    #     pipeline,
    #     genome=genome,
    #     current_segment_count=1,
    #     extra_segments=1,
    #     arm_count=5,
    # )

    state, best_genome = pipeline.auto_run(state)
    print("Evolution completed successfully")
    if best_genome is not None:
        save_genome(best_genome, output_dir="./models")

    return state, best_genome


if __name__ == "__main__":
    train_neat_controller()
