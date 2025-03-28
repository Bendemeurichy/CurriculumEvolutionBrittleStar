
import jax
import jax.numpy as jnp
from NEAT.observations import (
    get_joint_positions,
    get_direction_to_target,
    get_disk_direction
)
import NEAT.config as config
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG


def scale_actions(actions):
    """Scale actions to match joint limits"""
    lower_bounds = jnp.array([-1.047, -0.785] * sum(config.NUM_SEGMENTS_PER_ARM))
    upper_bounds = jnp.array([1.047, 0.785] * sum(config.NUM_SEGMENTS_PER_ARM))

    # Normalize between -1 and 1
    normalized_action = jnp.tanh(actions)

    action_ranges = (upper_bounds - lower_bounds) / 2
    action_midpoints = (upper_bounds + lower_bounds) / 2
    scaled_action = normalized_action * action_ranges + action_midpoints

    return scaled_action


def get_observation(env_state):
    """Extract observation from environment state"""
    joint_positions = []
    for arm in range(config.NUM_ARMS):
        joint_positions.append(get_joint_positions(env_state, arm))

    direction_to_target = get_direction_to_target(env_state)
    #print(direction_to_target,env_state.observations["disk_rotation"])
    # disk_direction = get_disk_direction(env_state)
    joint_positions_combined = jnp.concatenate(joint_positions)
    obs = jnp.concatenate([joint_positions_combined, direction_to_target])

    return obs

def get_environment_dims(env, state):
    """Get the input and output dimensions from the environment"""
    # Calculate number of inputs and outputs
    num_inputs = sum(
        [len(get_joint_positions(state, arm)) for arm in range(config.NUM_ARMS)]
    ) + len(get_direction_to_target(state))
    num_outputs = len(env.action_space.sample(rng=jax.random.PRNGKey(seed=0)))
    return num_inputs, num_outputs


def init_pipeline(problem):
    num_inputs, num_outputs = problem._input_dims, problem._output_dims
    return Pipeline(
        algorithm=NEAT(
            pop_size=config.POPULATION_SIZE,
            species_size=config.SPECIES_SIZE,
            survival_threshold=0.2,
            genome=DefaultGenome(
                num_inputs=num_inputs,
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
        seed=config.SEED,
        is_save=False,
        save_dir="output",
    )