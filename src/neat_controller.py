import os
import neat
import numpy as np
import pickle
import matplotlib.pyplot as plt
from environment import initialize_simulation, run_simulation_with_controller
import render
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio

#  Parameters
NUM_GENERATIONS = 50
NUM_ARMS = 4
NUM_SEGMENTS_PER_ARM = 2
SIMULATION_DURATION = 10  # seconds - increased from 5 to 10
VISUALIZE_TRAINING = True  # Set to True to visualize training progress
SAVE_BEST_EVERY_GEN = 5  # Save the best performing genome every N generations
TIME_SCALE = 1  # Slowed down from default of 2 for better visualization
FRAME_SKIP = 2  # Capture more frames (every 2nd frame instead of every 5th)
TARGET_DISTANCE = 3.0      # Default distance of the target
SUCCESS_THRESHOLD = 0.75   # How close to target (as a fraction of target_distance) counts as success
                           # Higher values = harder (must get closer to target)

os.makedirs("training_progress", exist_ok=True)

best_fitness_history = []
avg_fitness_history = []
generation_count = 0
best_distance_improvement = -float('inf')
best_genome_per_generation = {}

def eval_genome(genome, config):
    """Evaluate a single genome"""
    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Initialize the environment with custom simulation time and time scale
    env, state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        backend="MJC",
        simulation_time=SIMULATION_DURATION,
        time_scale=TIME_SCALE,
        target_distance=TARGET_DISTANCE,
    )
    
    # Track fitness metrics
    initial_distance_to_target = state.observations['xy_distance_to_target'][0]
    frames = []
    
    # Track distances over time for more detailed fitness
    distances = [initial_distance_to_target]
    
    # Get initial frame
    frame = env.render(state=state)
    processed_frame = render.post_render(
        render_output=frame, environment_configuration=environment_configuration
    )
    frames.append(processed_frame)
    
    # Run simulation for the full duration regardless of reaching target
    steps = 0
    max_steps = int(SIMULATION_DURATION / (environment_configuration.time_scale * 0.002 * environment_configuration.num_physics_steps_per_control_step))
    
    while steps < max_steps:
        # Use neural network to determine action based on observations
        obs = np.concatenate([
            state.observations['joint_position'],
            state.observations['joint_velocity'],
            state.observations['disk_position'],
            state.observations['disk_linear_velocity'],
            state.observations['unit_xy_direction_to_target'],
            state.observations['xy_distance_to_target']
        ])
        
        # Get action from neural network
        action = net.activate(obs)
        action = np.array(action)  # Convert to numpy array
        
        # Take a step in the environment
        old_state = state  # Save previous state
        state = env.step(state=state, action=action)
        steps += 1
        
        # If state is terminated (reached target), continue the simulation but create a "virtual" state
        if state.terminated and steps < max_steps:
            # Debug information to understand why it thinks it reached the goal
            current_distance = state.observations['xy_distance_to_target'][0]
            print(f"WARNING: Environment reports reaching target at distance: {current_distance:.3f}")
            print(f"Success threshold would be: {TARGET_DISTANCE * (1 - SUCCESS_THRESHOLD):.3f}")
            print(f"Target position: {state.info['xy_target_position']}")
            print(f"Current position: {state.observations['disk_position'][:2]}")
            
            # Keep the last valid observations but prevent early termination
            distances.append(current_distance)
            
            # Force the simulation to continue despite reaching the goal
            state = old_state  # Use previous state to continue
            state.terminated = False  # Prevent termination
            state.truncated = False   # Prevent truncation
        else:
            # Normal case - just track the distance
            current_distance = state.observations['xy_distance_to_target'][0]
            distances.append(current_distance)
        
        # Collect frame more frequently
        if steps % FRAME_SKIP == 0:
            processed_frame = render.post_render(
                env.render(state=state), environment_configuration
            )
            frames.append(processed_frame)
            
        # Break if we've reached the maximum steps
        if steps >= max_steps:
            break
    
    # Calculate fitness based on closest approach to target
    min_distance = min(distances)
    distance_improvement = initial_distance_to_target - min_distance
    
    # Additional reward if it maintained proximity to target
    avg_final_distance = np.mean(distances[-20:]) if len(distances) >= 20 else np.mean(distances)
    stability_bonus = max(0, initial_distance_to_target - avg_final_distance) * 2
    
    # Bonus for consistent movement toward target
    movement_consistency = 0
    for i in range(1, len(distances)):
        if distances[i] < distances[i-1]:  # Moving toward target
            movement_consistency += 0.1
    
    # Calculate final fitness (prioritize moving toward target)
    fitness = (distance_improvement * 10.0) + stability_bonus + movement_consistency
    
    # Store frames and metrics for the best genomes 
    genome.frames = frames
    genome.distance_improvement = distance_improvement
    genome.min_distance = min_distance
    
    # Print extra information for every 10th genome
    # Fix the genome ID extraction
    genome_id = genome.key if hasattr(genome, 'key') else 0  # Extract ID from genome
    if genome_id % 10 == 0:
        # Use our custom success threshold instead of the one from the environment
        our_success_threshold = TARGET_DISTANCE * (1 - SUCCESS_THRESHOLD)
        reached_target = min_distance < our_success_threshold
        print(f"Genome {genome_id}: Initial dist: {initial_distance_to_target:.2f}, " 
              f"Min dist: {min_distance:.2f}, Improvement: {distance_improvement:.2f}, "
              f"Fitness: {fitness:.2f}, Reached target: {reached_target} (threshold: {our_success_threshold:.2f})")
    
    return fitness

def eval_genomes(genomes, config):
    """Evaluate all genomes - this is the function NEAT expects"""
    global best_fitness_history, avg_fitness_history, generation_count, best_distance_improvement, best_genome_per_generation
    
    fitnesses = []
    best_genome = None
    best_fitness = -float('inf')
    
    print(f"\nEvaluating {len(genomes)} genomes in generation {generation_count}")
    
    for i, (genome_id, genome) in enumerate(genomes):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(genomes)} genomes evaluated")
        
        # Store genome ID for later use
        genome.key = genome_id
        
        fitness = eval_genome(genome, config)
        genome.fitness = fitness
        fitnesses.append(fitness)
        
        # Track best genome in this generation
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome
    
    # Calculate statistics
    avg_fitness = np.mean(fitnesses)
    avg_fitness_history.append(avg_fitness)
    best_fitness_history.append(best_fitness)
    
    # Store best genome of this generation
    best_genome_per_generation[generation_count] = best_genome
    
    # Check if this is the best overall
    if best_genome.distance_improvement > best_distance_improvement:
        best_distance_improvement = best_genome.distance_improvement
        # Save best genome's video if it made significant progress
        if VISUALIZE_TRAINING and best_genome.distance_improvement > 0.2:
            save_path = "training_progress/best_gen.mp4"
            render.create_animation(best_genome.frames, output_path=save_path)
    
    # Create visualization of progress
    if VISUALIZE_TRAINING:
        # Create plot of fitness progress
        fig = create_fitness_plot(generation_count, avg_fitness_history, best_fitness_history)
        
        # Save the plot image
        plot_path = "training_progress/current_fitness_progress.png"
        fig.savefig(plot_path)
        plt.close(fig)
        
        # Save current best genome animation if needed
        if generation_count % SAVE_BEST_EVERY_GEN == 0 and best_genome.frames:
            save_path = "training_progress/gen_current_best.mp4"
            render.create_animation(best_genome.frames, output_path=save_path)
            print(f"Saved animation of best genome at generation {generation_count}")
            
    # Update generation counter
    generation_count += 1

def create_fitness_plot(generation, avg_fitness, best_fitness):
    """Create a plot showing fitness progress over generations"""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(generation + 1), avg_fitness, 'b-', label='Average Fitness')
    plt.plot(range(generation + 1), best_fitness, 'r-', label='Best Fitness')
    plt.grid(True)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'NEAT Training Progress - Generation {generation}')
    plt.legend()
    return fig

def create_training_visualization():
    """Create an animated visualization of the training progress"""
    print("Creating training progress visualization...")
    
    # Collect plot images
    plot_images = []
    for gen in range(generation_count):
        plot_path = f"training_progress/fitness_progress_gen_{gen}.png"
        if os.path.exists(plot_path):
            plot_images.append(imageio.imread(plot_path))
    
    # Create GIF from plot images
    if plot_images:
        gif_path = "training_progress/fitness_evolution.gif"
        imageio.mimsave(gif_path, plot_images, duration=0.5)
        print(f"Created GIF of training progress at {gif_path}")
    
    # Create video of best genomes evolution if available
    best_videos = []
    for gen in range(0, generation_count, SAVE_BEST_EVERY_GEN):
        video_path = f"training_progress/gen_{gen}_best.mp4"
        if os.path.exists(video_path):
            best_videos.append(video_path)
    
    if best_videos:
        print(f"Best genome videos saved at: {', '.join(best_videos)}")

def run_neat(config_path):
    """Run NEAT algorithm to evolve brittle star controllers"""
    global generation_count
    generation_count = 0
    
    # Reset tracking variables
    best_fitness_history.clear()
    avg_fitness_history.clear()
    best_genome_per_generation.clear()
    
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Create the population
    pop = neat.Population(config)
    
    # Add a reporter to show progress in the terminal
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Save checkpoints every 5 generations
    pop.add_reporter(neat.Checkpointer(5, filename_prefix='./checkpoints/neat-checkpoint-'))
    
    # Set a very high fitness threshold so training continues for full number of generations
    config.fitness_threshold = 1000000.0
    
    # Run for a specified number of generations
    winner = pop.run(eval_genomes, NUM_GENERATIONS)
    
    # Save the winner genome
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    print(f"Best genome:\n{winner}")
    
    # Create visualization of training progress
    if VISUALIZE_TRAINING:
        create_training_visualization()
    
    # Visualize the best genome
    output_dir = "best_simulation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save animation of best performer
    video_path = f"{output_dir}/best_brittle_star.mp4"
    render.create_animation(winner.frames, output_path=video_path)
    print(f"Saved best performance animation to {video_path}")
    
    # Visualize stats
    plt.figure(figsize=(10, 6))
    plt.plot(stats.get_fitness_mean(), label="Average Fitness")
    plt.plot(stats.get_fitness_stdev(), label="Std Dev")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_dir}/fitness_evolution.png")
    plt.close()
    
    # Try to play the video
    render.play_video(video_path)
    
    return winner, stats

def create_neat_config():
    """Create a basic NEAT config file if it doesn't exist"""
    config_path = 'neat_config.txt'
    
    # FOR NOW - always create a new config file
    # Because we need to test multiple configurations, we will always create a new config file
    #if not os.path.exists(config_path):
    # ----------------------------------------------- 
    
    # Calculate input and output sizes based on environment
    env, state, _ = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        target_distance=TARGET_DISTANCE,
    )  # Remove success_threshold from here
    
    # Calculate number of inputs and outputs
    num_inputs = (
        len(state.observations['joint_position']) +
        len(state.observations['joint_velocity']) +
        len(state.observations['disk_position']) + 
        len(state.observations['disk_linear_velocity']) +
        len(state.observations['unit_xy_direction_to_target']) +
        len(state.observations['xy_distance_to_target'])
    )
    
    num_outputs = len(env.action_space.sample())
    
    config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10.0
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh relu sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 4
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        
    with open(config_path, 'w') as f:
        f.write(config_text)
    
    print(f"Created NEAT config file with {num_inputs} inputs and {num_outputs} outputs")

    return config_path

def load_and_use_winner(winner_path, config_path):
    """Load a trained genome and use it to control the brittle star"""
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Load the winner genome
    with open(winner_path, 'rb') as f:
        winner = pickle.load(f)
    
    # Create neural network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Initialize environment with longer simulation time and slower time scale
    env, state, environment_configuration = initialize_simulation(
        env_type="directed_locomotion",
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        backend="MJC",
        simulation_time=SIMULATION_DURATION * 2,  # Even longer for final visualization
        time_scale=TIME_SCALE,
        target_distance=TARGET_DISTANCE,
    )  # Remove success_threshold from here
    
    # Run simulation using the run_simulation_with_controller function
    print("Running simulation with trained controller...")
    frames = run_simulation_with_controller(env, state, environment_configuration, controller=net)
    print(f"Simulation complete with {len(frames)} frames!")
    
    # Create animation
    output_dir = "neat_simulation"
    os.makedirs(output_dir, exist_ok=True)
    video_path = f"{output_dir}/neat_controlled_brittle_star.mp4"
    render.create_animation(frames, output_path=video_path)
    print(f"Saved NEAT controlled animation to {video_path}")
    
    # Try to play the video
    render.play_video(video_path)

def load_checkpoint(checkpoint_path, config_path=None):
    """Load a NEAT checkpoint and return the population"""
    if config_path is None:
        # Create a new config if one isn't provided
        config_path = create_neat_config()
    
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Load population from checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    
    # Update the config
    pop.config = config
    
    return pop, config

def continue_training_from_checkpoint(checkpoint_path, additional_generations=20):
    """Continue training from a checkpoint"""
    global generation_count
    
    # Load the checkpoint
    pop, config = load_checkpoint(checkpoint_path)
    
    # Get the generation number from the checkpoint filename
    try:
        # Extract generation number from the checkpoint filename (assuming format: neat-checkpoint-X)
        checkpoint_filename = os.path.basename(checkpoint_path)
        generation_num = int(checkpoint_filename.split('-')[-1])
        generation_count = generation_num + 1  # Start from the next generation
    except:
        # If unable to extract generation number, assume it's the next one
        generation_count = len(best_fitness_history) + 1
    
    print(f"Continuing training from generation {generation_count}")
    
    # Add reporters for ongoing training
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5, filename_prefix='neat-checkpoint-'))
    
    # Set a very high fitness threshold to ensure training continues
    config.fitness_threshold = 1000000.0
    
    # Continue training for the specified number of additional generations
    winner = pop.run(eval_genomes, additional_generations)
    
    # Save the winner genome
    with open('winner_continued.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    print(f"Best genome after continued training:\n{winner}")
    
    # Create visualization of training progress
    if VISUALIZE_TRAINING:
        create_training_visualization()
    
    # Visualize the best genome
    output_dir = "best_simulation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save animation of best performer
    video_path = f"{output_dir}/best_brittle_star_continued.mp4"
    render.create_animation(winner.frames, output_path=video_path)
    print(f"Saved best performance animation to {video_path}")
    
    # Try to play the video
    render.play_video(video_path)
    
    return winner, stats

if __name__ == "__main__":
    # Create or load config file
    config_path = create_neat_config()
    
    # Check if user wants to continue training from a checkpoint
    checkpoint_pattern = 'neat-checkpoint-*'
    checkpoints = sorted([f for f in os.listdir('./checkpoints') if f.startswith('neat-checkpoint-')])
    
    if checkpoints:
        print("Found checkpoints:")
        for i, checkpoint in enumerate(checkpoints):
            print(f"[{i}] {checkpoint}")
        
        checkpoint_choice = input("Enter checkpoint number to load (or 'n' to start fresh, 'w' to load winner): ")
        
        if checkpoint_choice.lower() == 'w':
            # Check for winner file
            if os.path.exists('winner.pkl'):
                load_and_use_winner('winner.pkl', config_path)
            else:
                print("No winner file found, starting fresh training.")
                winner, stats = run_neat(config_path)
        elif checkpoint_choice.lower() != 'n' and checkpoint_choice.isdigit() and int(checkpoint_choice) < len(checkpoints):
            # Continue from checkpoint
            checkpoint_path = checkpoints[int(checkpoint_choice)]
            additional_gens = int(input("How many additional generations to run? "))
            winner, stats = continue_training_from_checkpoint(checkpoint_path, additional_gens)
        else:
            # Start fresh training
            winner, stats = run_neat(config_path)
    else:
        # No checkpoints found, check for winner file
        winner_path = 'winner.pkl'
        if os.path.exists(winner_path):
            user_input = input(f"Found existing winner at {winner_path}. Use it? (y/n): ")
            if user_input.lower() == 'y':
                load_and_use_winner(winner_path, config_path)
            else:
                winner, stats = run_neat(config_path)
        else:
            # Run NEAT algorithm from scratch
            winner, stats = run_neat(config_path)
    
    # Show summary after training
    if VISUALIZE_TRAINING:
        print("\nTraining summary available in training_progress/ directory")
        print("You can compare the best performers from different generations")
