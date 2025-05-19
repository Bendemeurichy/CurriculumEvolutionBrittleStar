"""Controller for brittle star movement using NEAT neural networks."""
import jax
import jax.numpy as jnp
from NEAT.multi_genome_pipeline import MultiGenomePipeline
from NEAT.observations import (
    extract_joint_positions,
    calculate_direction_to_target,
)
import config as config
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG
from NEAT.target_utils import get_direction_to_closest_target


def scale_actions_to_joint_limits(actions, num_segments_per_arm=config.NUM_SEGMENTS_PER_ARM):
    """Scale network output actions to match joint limits.
    
    Args:
        actions: Raw actions from the neural network
        num_segments_per_arm: Number of segments per arm
        
    Returns:
        Actions scaled to joint limits
    """
    lower_bounds = jnp.array([-1.047, -0.785] * sum(num_segments_per_arm))
    upper_bounds = jnp.array([1.047, 0.785] * sum(num_segments_per_arm))

    normalized_action = jnp.tanh(actions)

    action_ranges = (upper_bounds - lower_bounds) / 2
    action_midpoints = (upper_bounds + lower_bounds) / 2
    scaled_action = normalized_action * action_ranges + action_midpoints

    return scaled_action


def extract_observation(env_state, target=None):
    """Extract observation features from environment state.
    
    Args:
        env_state: Current environment state
        target: Optional target position override
        
    Returns:
        Observation vector for the neural network
    """
    joint_positions = []
    for arm in range(config.NUM_ARMS):
        joint_positions.append(extract_joint_positions(env_state, arm))

    if target is not None:
        disk_position = env_state.observations["disk_position"][:2]
        disk_rotation = env_state.observations["disk_rotation"][2]
        direction_to_target = get_direction_to_closest_target(
            disk_position, disk_rotation, target
        )
    else:
        direction_to_target = calculate_direction_to_target(env_state)

    joint_positions_combined = jnp.concatenate(joint_positions)
    return jnp.concatenate([joint_positions_combined, direction_to_target])


def calculate_environment_dimensions(env, state):
    """Calculate the input and output dimensions for the neural network.
    
    Args:
        env: Simulation environment
        state: Initial environment state
        
    Returns:
        Tuple of (num_inputs, num_outputs)
    """
    num_inputs = len(extract_observation(state))
    num_outputs = len(env.action_space.sample(rng=jax.random.PRNGKey(seed=0)))
    return num_inputs, num_outputs


def calculate_network_size_limits(num_inputs, num_outputs):
    """Calculate maximum nodes and connections for the neural network.
    
    Args:
        num_inputs: Number of input nodes
        num_outputs: Number of output nodes
        
    Returns:
        Tuple of (max_nodes, max_connections)
    """
    max_nodes = max(50, (num_inputs * num_outputs) * 2)
    max_connections = max_nodes * 2
    return max_nodes, max_connections


def initialize_neat_pipeline(problem):
    """Initialize a NEAT evolution pipeline.
    
    Args:
        problem: NEAT problem definition
        
    Returns:
        Configured NEAT pipeline
    """
    num_inputs, num_outputs = problem._input_dims, problem._output_dims
    max_nodes, max_connections = calculate_network_size_limits(num_inputs, num_outputs)

    return MultiGenomePipeline(
        algorithm=NEAT(
            pop_size=config.POPULATION_SIZE,
            species_size=config.SPECIES_SIZE,
            survival_threshold=0.2,
            genome=DefaultGenome(
                num_inputs=num_inputs,
                max_nodes=max_nodes,
                max_conns=max_connections,
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
        generation_limit=config.NUM_GENERATIONS,
        fitness_target=10000,
        early_stop_distance=config.TARGET_REACHED_THRESHOLD,
        early_stop_patience=config.EARLY_STOPPING,
        seed=config.SEED,
        is_save=False,
        save_dir="output",
    )

