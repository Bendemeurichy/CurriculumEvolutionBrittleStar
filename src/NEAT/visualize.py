
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from neat_controller import scale_actions, get_observation
import NEAT.config as config
from environment import initialize_simulation
import render
import pickle
from NEAT.neat_controller import init_pipeline
from NEAT.neat_problem import BrittleStarEnv
import numpy as np
from NEAT.genome_extension import extend_genome, add_segment_to_genome

def create_animation(frames, output_path):
    """Create and save an animation from frames"""
    directory = os.path.dirname(output_path)
    if directory: 
        os.makedirs(directory, exist_ok=True)

    render.create_animation(frames, output_path=output_path)
    print(f"Saved animation to {output_path}")
    render.play_video(output_path)


def load_model(model_path):
    """Load a model from a pickle file"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def save_network_visualization(genome,segments, save_path=None):

    if save_path is None:
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "brittle_star_neural_network.svg")


    problem = BrittleStarEnv(num_segments_per_arm=segments)
    pipeline = init_pipeline(problem)
    state = pipeline.setup()


    network = pipeline.algorithm.genome.network_dict(state, *genome)
    pipeline.algorithm.genome.visualize(network, save_path=save_path)
    

def visualize_neural_network(model_path, save_path=None,segments=config.NUM_SEGMENTS_PER_ARM):
    genome = load_model(model_path)
    save_network_visualization(genome, save_path=save_path,segments=segments)
    

def visualize_model(model_path, save_path=None,segments=config.NUM_SEGMENTS_PER_ARM):

    genome = load_model(model_path)
    #genome = add_segment_to_genome(genome, 1)
    # save_network_visualization(genome)
    # exit(1)
    problem = BrittleStarEnv(num_segments_per_arm=segments)

    pipeline = init_pipeline(problem)
    state = pipeline.setup()

    
    visualize_brittlestar(state=state, genome=genome, algorithm=pipeline.algorithm, save_path=save_path,segments=segments)


def visualize_brittlestar(state, genome, algorithm,segments, save_path=None):
    """Visualize the trained brittlestar model"""
    print("Creating visualization...")

    env, env_state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=config.NUM_ARMS,
        num_segments_per_arm=config.NUM_SEGMENTS_PER_ARM,
        backend="MJC",
        simulation_time=config.SIMULATION_DURATION,
        time_scale=config.TIME_SCALE,
        target_distance=config.TARGET_DISTANCE,
        num_physics_steps_per_control_step=config.NUM_PHYSICS_STEPS_PER_CONTROL_STEP,
        seed=config.SEED,
    )



    if config.TARGET_POSITION is not None:
        env_state = env.reset(rng=env_state.rng, target_position=config.TARGET_POSITION)
    
    
    transformed_genome = algorithm.transform(state, genome)

    max_steps = config.MAX_STEPS_VISUALIZATION
    frames = []



    # t = env_state.mj_data.xpos[env_state.mj_model.body("target").id][:2]
    # target = [t,t]
    # print("Target position:", target)
    target = None
    obs = get_observation(env_state)

    initial_distance = env_state.observations["xy_distance_to_target"][0]


    min_distance = initial_distance
    total_reward = 0.0

    
    

    camera_id = 0
    env._env._mj_model.cam_pos[camera_id] *= 0.35  

    print("Running simulation...")
    for step in range(max_steps):
        frame = env.render(env_state)
        processed_frame = render.post_render(
            render_output=frame, environment_configuration=environment_configuration
        )
        frames.append(processed_frame)

        action = algorithm.forward(state, transformed_genome, obs)

        scaled_action = scale_actions(action, num_segments_per_arm=segments)

        #print("=>",env_state.mj_data.xpos[env_state.mj_model.body("target").id])
        # target_id = env_state.mj_model.body("target").id
        # target = env_state.mj_data.xpos[target_id]
        # jax.debug.print("({}, {})", target[0], target[1])

        env_state = env.step(state=env_state, action=scaled_action)
        # print(env_state.observations)
        obs = get_observation(env_state,target)
        

        current_distance = env_state.observations["xy_distance_to_target"][0]
        min_distance = min(float(min_distance), float(current_distance))

        reward = current_distance
        total_reward += reward

        if current_distance < 0.1:
            print("Target reached!")
            break

        print(
            f"Step {step}: Distance = {current_distance:.4f}, Min Distance = {min_distance:.4f}"
        )

    # Display results
    print("\n=== Visualization Results ===")
    print(f"Initial distance: {initial_distance:.4f}")
    print(f"Minimum distance: {min_distance:.4f}")
    print(f"Distance improvement: {initial_distance - min_distance:.4f}")
    print(f"Total reward: {total_reward:.4f}")

    if save_path is None:
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "brittlestar_visualization.mp4")
    else:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    create_animation(frames, save_path)

    return {
        "frames": frames,
        "fitness": initial_distance - min_distance,
        "initial_distance": initial_distance,
        "min_distance": min_distance,
        "distance_improvement": initial_distance - min_distance,
        "total_reward": total_reward,
    }



if __name__ == "__main__":
    model_filename = "best_8_genome_4_seg.pkl"
    model_path = os.path.join(os.path.dirname(__file__), "../models/curr_test_6", model_filename)
    segments = config.NUM_SEGMENTS_PER_ARM

    parts = model_filename.split("_")
    if len(parts) == 5:
        num_segments = int(parts[-2])
        segments = [num_segments] * config.NUM_ARMS
        config.NUM_SEGMENTS_PER_ARM = segments

    visualize_model(model_path=model_path,segments=segments)
    #visualize_neural_network(model_path=model_path,segments=segments)
    
