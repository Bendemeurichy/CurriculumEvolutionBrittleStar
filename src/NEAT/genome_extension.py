import copy
import jax.numpy as jnp
import numpy as np  # For random number generation
from tensorneat.common import State
from tensorneat.genome.utils import add_conn, add_node
from tensorneat.pipeline import Pipeline
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from NEAT.neat_controller import init_pipeline
from NEAT.neat_problem import BrittleStarEnv
from NEAT.visualize import load_model


### CODE FOR EXTENDING AN EXISTING GENOME INSTANCE ###


def extend_genome(
    state: State,
    pipeline: Pipeline,
    genome: tuple[jnp.ndarray, jnp.ndarray],
    extra_inputs: int = 0,
    extra_outputs: int = 0,
) -> State:
    """Extend the existing genome with additional inputs and outputs.
    This function creates a new genome with the same structure as the original one,
    but with additional nodes and connections to accommodate the extra inputs and outputs.
    The new genome is then transformed into a new state object.

    Args:
        state (State): State of the NEAT algorithm. (pass to the pipeline)
        genome (tuple[jnp.ndarray, jnp.ndarray]): The result of a previous NEAT run. (nodes, conns)
        extra_inputs (int, optional): The amount of extra inputs wanted. Defaults to 0.
        extra_outputs (int, optional): The amount of extra outputs wanted. Defaults to 0.

    Returns:
        State: The new state object with the extended genome.
    """
    old_output_start = pipeline.algorithm.genome.output_idx[0] - extra_inputs
    new_output_start = old_output_start + extra_inputs
    output_amount = len(pipeline.algorithm.genome.output_idx)

    nodes, conns = shift_index_outputs(
        genome[0], genome[1], old_output_start, new_output_start
    )

    pop = pipeline.algorithm.ask(state)  # Retrieve population
    all_nodes = pop[0]
    all_conns = pop[1]

    for instance in range(len(all_nodes)):
        all_nodes[instance] = nodes
        # TODO ADD THE NEW INPUTS AND OUTPUTS TO THE GENOME
        for i in range(extra_inputs, 0, -1):
            new_idx = new_output_start - i
            attributes = [
                new_idx,  # Node ID
                np.random.normal(0, 0.5),  # bias
                0,  # aggregation function
                np.random.randint(0, 3),  # activation (random from 0 to 2)
            ]
            all_nodes[instance] = add_node(all_nodes[instance], attributes)
        for i in range(extra_outputs):
            new_idx = new_output_start + output_amount + i
            attributes = [
                new_idx,  # Node ID
                np.random.normal(0, 0.5),  # bias
                0,  # aggregation function
                np.random.randint(0, 3),  # activation (random from 0 to 2)
            ]
            all_nodes[instance] = add_node(all_nodes[instance], attributes)

    for instance in range(len(all_conns)):
        all_conns[instance] = conns
        # TODO CONNECTIONS FROM NEW INPUTS TO ALL NON-INPUT NODES AND FROM NON-OUTPUT NODES TO NEW OUTPUTS

    pop = all_nodes, all_conns
    state = pipeline.algorithm.tell(state, pop)  # Update state with population

    return pipeline, state


def shift_index_outputs(nodes, connections, old_output_start, new_output_start):
    """Shift the index of the output nodes in the genome to match the new output start index.
        Do the same for the connections.
    Args:
        nodes (np.ndarray): The nodes of the genome.
        connections (np.ndarray): The connections of the genome.
        old_output_start (int): The old output start index.
        new_output_start (int): The new output start index.

    Returns:
        np.ndarray: The updated nodes and connections with shifted output indices.
    """
    diff = new_output_start - old_output_start

    # Shift output indices upwards
    for i in range(old_output_start, nodes.shape[0]):
        if nodes[i, 0] == jnp.nan:
            break
        nodes[i, 0] += diff

    # Shift connection indices
    for i in range(connections.shape[0]):
        if connections[i, 0] == jnp.nan:
            break
        if connections[i, 1] >= old_output_start:
            connections[i, 1] += diff
        if connections[i, 0] >= old_output_start:
            connections[i, 0] += diff
    return nodes, connections
