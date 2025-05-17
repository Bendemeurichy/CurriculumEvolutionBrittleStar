import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from NEAT.genome_extension import extend_genome
from NEAT.neat_problem import BrittleStarEnv
from NEAT.neat_controller import initialize_neat_pipeline
from NEAT.visualize import load_model
import NEAT.config as config


def save_genome(best, output_dir="./", filename="best_genome.pkl"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, filename)
    with open(model_path, "wb") as f:
        pickle.dump(best, f)
    print(f"Best genome saved to {model_path}")


def train_neat_controller(extend_genome=True):
    """Train a TensorNEAT controller for the brittle star"""
    print("Starting NEAT training for brittle star locomotion...")

    problem = BrittleStarEnv()  # Use the new RLEnv-based class
    # Get input and output dimensions
    num_inputs, num_outputs = problem._input_dims, problem._output_dims
    print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

    pipeline = initialize_neat_pipeline(problem)

    print("Initializing TensorNEAT state...")
    state = pipeline.setup()

    if extend_genome:
        genome = load_model("./models/genome_2_seg_rand.pkl")

        state = extend_genome(
            state,
            pipeline,
            genomes=[genome],
            current_segment_count=2,
            extra_segments=1,
            arm_count=5,
        )

    state, best_genomes, generations = pipeline.auto_run(state)
    print(f"Top {len(best_genomes)} genomes:")
    print("Evolution completed successfully")
    if best_genomes is not None:
        save_genome(best_genomes, output_dir="./models")

    return state, best_genomes


def train_neat_curriculum(start_genome=None):

    start_num_segments = 1
    if start_genome is not None:
        genome = load_model(start_genome)
        start_num_segments = 3

    config.NUM_SEGMENTS_PER_ARM = [start_num_segments] * config.NUM_ARMS
    problem = BrittleStarEnv()

    pipeline = initialize_neat_pipeline(problem)
    state = pipeline.setup()

    if start_genome is not None:
        state = extend_genome(
            state,
            pipeline,
            genomes=[genome],
            current_segment_count=start_num_segments - 1,
            extra_segments=1,
            arm_count=config.NUM_ARMS,
        )

    generations_per_segment = []

    for i in range(start_num_segments, 6):
        print(
            f"Starting NEAT training for brittle star locomotion with {i} segments..."
        )

        num_inputs, num_outputs = problem._input_dims, problem._output_dims
        print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

        state, best_genomes, generations = pipeline.auto_run(state)

        generations_per_segment.append(generations)

        for j, genome in enumerate(best_genomes):
            save_genome(
                genome, output_dir="./models", filename=f"best_{j}_genome_{i}_seg.pkl"
            )

        config.NUM_SEGMENTS_PER_ARM = [i + 1] * config.NUM_ARMS

        print(
            f"Updating the number of segments per arm to {config.NUM_SEGMENTS_PER_ARM}"
        )
        problem = BrittleStarEnv(num_segments_per_arm=[i + 1] * config.NUM_ARMS)
        pipeline = initialize_neat_pipeline(problem)
        state = pipeline.setup()
        print("Initializing TensorNEAT state...")
        state = extend_genome(
            state,
            pipeline,
            genomes=best_genomes,
            current_segment_count=i,
            extra_segments=1,
            arm_count=config.NUM_ARMS,
        )

    for i, generations in enumerate(generations_per_segment):
        print(
            f"Training for {i + 1} segments took {generations} generations to reach the target"
        )

    print(
        f"Evolution completed successfully, total generations: {sum(generations_per_segment)}"
    )


def train_neat_no_curriculum():
    config.NUM_SEGMENTS_PER_ARM = [1] * config.NUM_ARMS
    problem = BrittleStarEnv()
    pipeline = initialize_neat_pipeline(problem)
    state = pipeline.setup()

    generations_per_segment = []

    for i in range(1, 6):
        print(
            f"Starting NEAT training for brittle star locomotion with {i} segments..."
        )

        num_inputs, num_outputs = problem._input_dims, problem._output_dims
        print(f"Environment requires {num_inputs} inputs and {num_outputs} outputs")

        state, best_genomes, generations = pipeline.auto_run(state)

        generations_per_segment.append(generations)

        for j, genome in enumerate(best_genomes):
            save_genome(
                genome,
                output_dir="./models",
                filename=f"best_{j}_genome_{i}_seg_direct.pkl",
            )

        config.NUM_SEGMENTS_PER_ARM = [i + 1] * config.NUM_ARMS
        print(
            f"Updating the number of segments per arm to {config.NUM_SEGMENTS_PER_ARM}"
        )
        problem = BrittleStarEnv(num_segments_per_arm=[i + 1] * config.NUM_ARMS)
        pipeline = initialize_neat_pipeline(problem)
        state = pipeline.setup()
        print("Initializing TensorNEAT state...")

    for i, generations in enumerate(generations_per_segment):
        print(
            f"Training for {i + 1} segments took {generations} generations to reach the target"
        )

    print(
        f"Evolution completed successfully, total generations: {sum(generations_per_segment)}"
    )


if __name__ == "__main__":
    import argparse

    # Arguments for training mode
    parser = argparse.ArgumentParser(
        description="Train NEAT controller for brittle star locomotion."
    )
    parser.add_argument(
        "--mode",
        choices=["curriculum", "no_curriculum"],
        default="no_curriculum",
        help="Choose training mode: curriculum or no_curriculum",
    )

    args = parser.parse_args()

    if args.mode == "curriculum":
        train_neat_curriculum()
    else:
        train_neat_no_curriculum()
