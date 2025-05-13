
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import render

import NEAT.config as config
from environment import initialize_simulation
from neat_controller import scale_actions, get_observation
from NEAT.visualize import load_model 
from NEAT.neat_controller import init_pipeline
from NEAT.neat_problem import BrittleStarEnv


def run_model(model_path,segments = config.NUM_SEGMENTS_PER_ARM):

    genome = load_model(model_path)
    #genome = add_segment_to_genome(genome, 1)
    # save_network_visualization(genome)
    # exit(1)
    problem = BrittleStarEnv(num_segments_per_arm=segments)

    pipeline = init_pipeline(problem)
    state = pipeline.setup()

    
    return get_step_count(state=state, genome=genome, algorithm=pipeline.algorithm,segments=segments)



def get_step_count(state, genome, algorithm,segments):
    env, env_state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=config.NUM_ARMS,
        num_segments_per_arm=segments,
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
        obs = get_observation(env_state,targets=target)
        

        current_distance = env_state.observations["xy_distance_to_target"][0]
        min_distance = min(float(min_distance), float(current_distance))

        reward = current_distance
        total_reward += reward

        if current_distance < 0.1:
            print("Target reached! after", step, "steps")
            break

        # print(
        #     f"Step {step}: Distance = {current_distance:.4f}, Min Distance = {min_distance:.4f}"
        # )



    return step

def find_best_genomes_by_segments(folder: str):
    """Find the best genome for each segment configuration in the given folder"""
    best_genomes = {}  # Map of num_segments -> (model_path, step_count)

    for filename in os.listdir(folder):
        if filename.endswith(".pkl") and "genome" in filename:
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


if __name__ == "__main__":
    # Example usage
    folder_path = os.path.join(os.path.dirname(__file__), "../models/curr_test_2")

    best_genomes = find_best_genomes_by_segments(folder_path)
    
    for segments, model_path, step_count in best_genomes:
        print(f"Segments: {segments}, Model Path: {model_path}, Step Count: {step_count}")