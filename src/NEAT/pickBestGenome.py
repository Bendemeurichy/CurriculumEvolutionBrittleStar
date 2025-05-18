import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import render
import jax
import jax.numpy as jnp
from functools import lru_cache
from typing import Dict, Tuple, Any

import NEAT.config as config
from environment import initialize_simulation
from neat_controller import scale_actions_to_joint_limits, extract_observation
from NEAT.visualize import load_model 
from NEAT.neat_controller import initialize_neat_pipeline
from NEAT.neat_problem import BrittleStarEnv
from initialize import mujoco

# Cache for environments and pipelines
# Structure: {num_segments: (problem, pipeline, state, env, env_state, env_config)}
ENV_CACHE: Dict[int, Tuple[Any, Any, Any, Any, Any, Any]] = {}


config.TARGET_POSITION = (-0.6930648989168606, -1.8760759701806753)

def run_model(model_path,segments = config.NUM_SEGMENTS_PER_ARM):

    genome = load_model(model_path)
    
    # Get cached environment or create a new one
    problem, pipeline, state, env, env_state, _ = get_cached_environment(segments)
    
    # Reset environment state for new run
    if config.TARGET_POSITION is not None:
        env_state = env.reset(rng=env_state.rng, target_position=config.TARGET_POSITION)
    else:
        env_state = env.reset(rng=env_state.rng)
    
    return get_step_count(
        state=state, 
        genome=genome, 
        algorithm=pipeline.algorithm,
        segments=segments,
        env=env,
        env_state=env_state
    )



def get_step_count(state, genome, algorithm, segments, env, env_state):
    # Setup JAX to use GPU if available
    key = jax.random.PRNGKey(config.SEED)

    

    if config.TARGET_POSITION is not None:
        env_state = env.reset(rng=env_state.rng, target_position=config.TARGET_POSITION)
    
    # JIT-compile the forward function for better performance
    transformed_genome = algorithm.transform(state, genome)
    
    # JIT-compile the step function
    @jax.jit
    def step_env(env_state, obs):
        action = algorithm.forward(state, transformed_genome, obs)
        scaled_action = scale_actions_to_joint_limits(action, num_segments_per_arm=segments)
        next_env_state = env.step(state=env_state, action=scaled_action)
        next_obs = extract_observation(next_env_state)
        return next_env_state, next_obs, next_env_state.observations["xy_distance_to_target"][0]

    max_steps = config.MAX_STEPS_VISUALIZATION

    # Initialize with JAX arrays
    obs = extract_observation(env_state)
    initial_distance = env_state.observations["xy_distance_to_target"][0]
    min_distance = jnp.array(initial_distance)
    
    # Use mutable variables only for tracking since JAX prefers immutable state
    min_distance_val = float(initial_distance)

    # Camera setup (non-jax operation)
    camera_id = 0
    env._env._mj_model.cam_pos[camera_id] *= 0.35 

    print("Running simulation on", jax.devices()[0])
    
    # Main simulation loop
    for step in range(max_steps):
        # Use JIT-compiled step function
        env_state, obs, current_distance = step_env(env_state, obs)
        
        # Update minimum distance (non-JAX operation)
        min_distance_val = min(min_distance_val, float(current_distance))

        # Terminal condition check
        if float(current_distance) < config.TARGET_REACHED_THRESHOLD:
            print("Target reached! after", step, "steps")
            break

    return step

def find_best_genomes_by_segments(folder: str):
    """Find the best genome for each segment configuration in the given folder"""
    best_genomes = {}  # Map of num_segments -> (model_path, step_count)

    for filename in os.listdir(folder):
        if filename.endswith(".pkl") and "genome" in filename and "_direct" not in filename:
            print(f"Processing file: {filename}")
            # Parse the filename to extract segment count
            # Format: best_0_genome_2_seg
            parts = filename.split("_")
            if len(parts) >= 4 and parts[-1] == "seg.pkl":
                num_segments = int(parts[-2])
                segments = [num_segments] * config.NUM_ARMS
                config.NUM_SEGMENTS_PER_ARM = segments
                model_path = os.path.join(folder, filename)
                print(f"Running model with {num_segments} segments: {model_path}")
                step_count = run_model(model_path, segments)
                print(f"Model path: {model_path}, Step count: {step_count}")

                # Lower step count is better (reached target faster)
                if num_segments not in best_genomes or step_count < best_genomes[num_segments][1]:
                    best_genomes[num_segments] = (model_path, step_count)
        
    # Format the results as a list of (segments, model_path, step_count)
    result = [(segments, path, steps) for segments, (path, steps) in best_genomes.items()]
    result.sort(key=lambda x: x[0])  # Sort by number of segments
    
    return result

def get_cached_environment(segments):
    """
    Get or create a cached environment, pipeline and state for the given segment count
    
    Args:
        segments: Number of segments per arm
        
    Returns:
        Tuple of (problem, pipeline, state, env, env_state, environment_configuration)
    """
    # Check if we already have a cached environment for this segment count
    if segments[0] in ENV_CACHE:
        print(f"Using cached environment for {segments} segments")
        return ENV_CACHE[segments[0]]
    
    print(f"Initializing new environment for {segments} segments")
    
    # Create a new environment
    problem = BrittleStarEnv(num_segments_per_arm=segments)
    pipeline = initialize_neat_pipeline(problem)
    state = pipeline.setup()
    
    env, env_state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=config.NUM_ARMS,
        num_segments_per_arm=segments,
        backend="MJX",
        simulation_time=config.SIMULATION_DURATION,
        time_scale=config.TIME_SCALE,
        target_distance=config.TARGET_DISTANCE,
        num_physics_steps_per_control_step=config.NUM_PHYSICS_STEPS_PER_CONTROL_STEP,
        seed=config.SEED,
    )
    
    # Cache the environment
    ENV_CACHE[segments[0]] = (problem, pipeline, state, env, env_state, environment_configuration)
    
    return ENV_CACHE[segments[0]] 


if __name__ == "__main__":
    # Setup JAX to use GPU if available
    
    print("JAX devices available:", jax.devices())
    
    # Example usage
    folder_path = os.path.join(os.path.dirname(__file__), "../models/final")

    best_genomes = find_best_genomes_by_segments(folder_path)
    
    for segments, model_path, step_count in best_genomes:
        print(f"Segments: {segments}, Model Path: {model_path}, Step Count: {step_count}")