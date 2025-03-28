


# General simulation parameters
NUM_GENERATIONS = 20
NUM_ARMS = 5
NUM_SEGMENTS_PER_ARM = [2, 0, 0, 2, 0]  # Number of segments per arm
SIMULATION_DURATION = 40  # Simulation duration in seconds

MAX_STEPS_TRAINING = 400  # Maximum steps for training
MAX_STEPS_VISUALIZATION = 400  # Maximum steps for visualization

TIME_SCALE = 2  # Time scale for simulation
TARGET_DISTANCE = 3.0  # Target distance for locomotion

# Visualization and saving
VISUALIZE_TRAINING = True  # Whether to visualize training
NUM_PHYSICS_STEPS_PER_CONTROL_STEP=10 # How many physics steps per action

# NEAT algorithm parameters
POPULATION_SIZE = 100  # Population size for NEAT
SPECIES_SIZE = 5  # Number of species in the population

SEED=50