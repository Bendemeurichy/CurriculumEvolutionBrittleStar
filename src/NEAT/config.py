


# General simulation parameters
NUM_GENERATIONS = 100
NUM_ARMS = 5
NUM_SEGMENTS_PER_ARM = 1  # Number of segments per arm
SIMULATION_DURATION = 40  # Simulation duration in seconds

MAX_STEPS_TRAINING = 400  # Maximum steps for training
MAX_STEPS_VISUALIZATION = 400  # Maximum steps for visualization

TIME_SCALE = 2  # Time scale for simulation
TARGET_DISTANCE = 2.0  # Target distance for locomotion (is ignored when TARGET_POSITION is set)
TARGET_POSITION = None #[0.43104,-2.96887,0] # None OR select a specific target position

# Visualization and saving
NUM_PHYSICS_STEPS_PER_CONTROL_STEP=10 # How many physics steps per action

# NEAT algorithm parameters
POPULATION_SIZE = 50  # Population size for NEAT | the overall number of genomes (neural networks) in your entire population that will evolve over generations.
SPECIES_SIZE = 10  # Number of species in the population | number of species the algorithm should maintain.

SEED=52


# Brittle star config
DISK_DIAMETER = 0.25           # DEFAULT: 0.25
DISK_HEIGHT = 0.025             # DEFAULT: 0.025
START_SEGMENT_RADIUS = 0.025    # DEFAULT: 0.025
STOP_SEGMENT_RADIUS = 0.0125    # DEFAULT: 0.0125
START_SEGMENT_LENGTH = 0.075    # DEFAULT: 0.075
STOP_SEGMENT_LENGTH = 0.025    # DEFAULT: 0.025


from jax import numpy as jnp
if isinstance(NUM_SEGMENTS_PER_ARM, int):
    NUM_SEGMENTS_PER_ARM = [NUM_SEGMENTS_PER_ARM] * NUM_ARMS
if TARGET_POSITION is not None:
    TARGET_POSITION = jnp.array(TARGET_POSITION)