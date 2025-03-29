


# General simulation parameters
NUM_GENERATIONS = 100
NUM_ARMS = 5
NUM_SEGMENTS_PER_ARM = [1,0,0,1,0]  # Number of segments per arm
SIMULATION_DURATION = 40  # Simulation duration in seconds

MAX_STEPS_TRAINING = 400  # Maximum steps for training
MAX_STEPS_VISUALIZATION = 400  # Maximum steps for visualization

TIME_SCALE = 2  # Time scale for simulation
TARGET_DISTANCE = 3.0  # Target distance for locomotion

# Visualization and saving
NUM_PHYSICS_STEPS_PER_CONTROL_STEP=10 # How many physics steps per action


# NEAT algorithm parameters
POPULATION_SIZE = 100  # Population size for NEAT
SPECIES_SIZE = 10  # Number of species in the population

SEED=50


# Brittle star config
DISK_DIAMETER = 0.25           # DEFAULT: 0.25
DISK_HEIGHT = 0.025             # DEFAULT: 0.025
START_SEGMENT_RADIUS = 0.025    # DEFAULT: 0.025
STOP_SEGMENT_RADIUS = 0.0125    # DEFAULT: 0.0125
START_SEGMENT_LENGTH = 0.075    # DEFAULT: 0.075
STOP_SEGMENT_LENGTH = 0.025    # DEFAULT: 0.025



if isinstance(NUM_SEGMENTS_PER_ARM, int):
    NUM_SEGMENTS_PER_ARM = [NUM_SEGMENTS_PER_ARM] * NUM_ARMS